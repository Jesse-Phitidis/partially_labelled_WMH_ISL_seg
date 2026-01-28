from pathlib import Path
import nibabel as nib
import torchio as tio
import subprocess
from tempfile import TemporaryDirectory
from torchio.data.io import sitk_to_nib, nib_to_sitk
import numpy as np
import SimpleITK as sitk

multiclass_label_presedence = ["ISLtotal", "WMH"]

def process_subject(
    images: dict[str, dict], 
    labels: dict[str, dict],
    target: tuple | str,
    image_interp: str,
    label_interp: str,
    min_shape: tuple[int] | None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:

    """A function to preprocess a subject.

    Args:
        images: A dictionary with keys as image names (e.g. FLAIR) and values as a dictionary which must 
            contain a key 'nii' with a value of type nibabel.Nifti1image.
        labels: Same as images.
        target: A tuple of voxel sizes (e.g. (1,1,1)) or an image or label name.
        image_interp: The type of image interpolation for use with torchio.Resample
        label_interp: The same as image_interp, but must be nearest or linear. If linear is used then ResampleSegLinear 
            function is used instead of torchio.Resample.
        min_shape: The minimum allowable shape in each dimension. 

    Returns:
        The images and labels
    """

    # Check images are in the same space
    first_image = next(iter(images.values()))["nii"]
    for val_dict in (images | labels).values():
        assert np.array_equal(first_image.shape, val_dict["nii"].shape), "Not all images have the same shape before processing"
        assert np.allclose(first_image.affine, val_dict["nii"].affine, atol=1e-5), "Not all affines are the same before processing"

    # Instantiate transforms
    to_canonical = tio.ToCanonical()
    if isinstance(target, str):
        shape = (images | labels)[target]["nii"].shape
        affine = (images | labels)[target]["nii"].affine
        target = (shape, affine)
    resample_image = tio.Resample(target=target, image_interpolation=image_interp)
    assert label_interp in ["nearest", "linear"], "label_interp must be 'nearest' or 'linear'"
    if label_interp == "nearest":
        resample_label = tio.Resample(target=target, image_interpolation=label_interp)
    if label_interp == "linear":
        resample_label = ResampleSegLinear(target=target)
    if min_shape is not None:
        ensure_shape = EnsureShapeAtLeastTransform(min_shape=min_shape)
    
    # To canonical
    for val_dict in images.values():
        val_dict["nii"] = to_canonical(val_dict["nii"])
    for val_dict in labels.values():
        val_dict["nii"] = to_canonical(val_dict["nii"])

    # Bias field correction (sitk N4)
    for val_dict in images.values():
        val_dict["nii"] = bias_field_correction(val_dict["nii"])

    # Resample
    for val_dict in images.values():
        val_dict["nii"] = resample_image(val_dict["nii"])
    for val_dict in labels.values():
        val_dict["nii"] = resample_label(val_dict["nii"])

    # Brain extraction (SynthStrip)
    for val_dict in images.values():
        val_dict["nii"] = brain_extraction(val_dict["nii"])

    # Crop to brain
    min_indices, max_indices = [], []
    for val_dict in images.values():
        min, max = get_min_max_nonzero_indices(val_dict["nii"])
        min_indices.append(min), max_indices.append(max)
    min_indices = np.min(np.stack(min_indices, axis=0), axis=0)
    max_indices = np.max(np.stack(max_indices, axis=0), axis=0)
    shape = next(iter(images.values()))["nii"].shape
    crop = get_crop_transform(min_indices, max_indices, shape)
    for val_dict in images.values():
        val_dict["nii"] = crop(val_dict["nii"])
    for val_dict in labels.values():
        val_dict["nii"] = crop(val_dict["nii"])

    # Ensure shape at least
    if min_shape is not None:
        for val_dict in images.values():
            val_dict["nii"] = ensure_shape(val_dict["nii"])
        for val_dict in labels.values():
            val_dict["nii"] = ensure_shape(val_dict["nii"])
            
    # Check images are in the same space
    first_image = next(iter(images.values()))["nii"]
    for val_dict in (images | labels).values():
        assert np.array_equal(first_image.shape, val_dict["nii"].shape), "Not all images have the same shape after processing"
        assert np.allclose(first_image.affine, val_dict["nii"].affine, atol=1e-5), "Not all affines are the same after processing"
        val_dict["nii"] = nib.Nifti1Image(val_dict["nii"].get_fdata(), first_image.affine, val_dict["nii"].header) # Avoid problems with small diffs

    return images, labels


def bias_field_correction(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    data, affine, header = nii.get_fdata(), nii.affine, nii.header
    sitk_image = nib_to_sitk(data=data[None,...], affine=affine)
    sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1)
    sitk_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_mask)
    data, _ = sitk_to_nib(sitk_image)
    nii = nib.Nifti1Image(data[0,...], affine, header)
    return nii


def brain_extraction(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """ This function assumes freesurfer is set up correctly"""
    tempdir = TemporaryDirectory()
    nii_in_path = Path(tempdir.name) / "in.nii.gz"
    nii_out_path = Path(tempdir.name) / "out.nii.gz"
    nib.save(nii, nii_in_path)
    subprocess.run(["mri_synthstrip", "-i", str(nii_in_path), "-o", str(nii_out_path)], stdout=subprocess.DEVNULL)
    nii = nib_load(nii_out_path, lazy=False)
    return nii


class ResampleSegLinear(tio.Transform):
    
    def __init__(self, target=(1, 1, 1)):
        super().__init__(parse_input=False)
        self.resampler = tio.Resample(target=target, image_interpolation="linear")
        
    def apply_transform(self, im: nib.Nifti1Image) -> nib.Nifti1Image:
        data = im.get_fdata()
        unique = np.unique(data)
        label_to_index = {value: index for index, value in enumerate(unique)}
        data_index = np.zeros_like(data)
        for original_label, index in label_to_index.items():
            data_index[data == original_label] = index
        data_one_hot = np.eye(len(unique))[data_index.astype(int)].transpose(3, 0, 1, 2)
        images_resampled = []
        for channel in range(len(unique)):
            channel_im = nib.Nifti1Image(data_one_hot[channel], affine=im.affine, header=im.header)
            channel_im_resampled = self.resampler(channel_im)
            images_resampled.append(channel_im_resampled)
        out_affine, out_header = images_resampled[0].affine, images_resampled[0].header
        arrays_resampled = [image.get_fdata() for image in images_resampled]
        channel_array_resampled = np.stack(arrays_resampled, axis=0)
        result_array_indexes = np.argmax(channel_array_resampled, axis=0)
        result_array_labels = np.zeros_like(result_array_indexes)
        for original_label, index in label_to_index.items():
            result_array_labels[result_array_indexes == index] = original_label
        im = nib.Nifti1Image(result_array_labels, affine=out_affine, header=out_header)
        return im
    

def get_min_max_nonzero_indices(nii: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    data = nii.get_fdata()
    data_nonzero = np.where(data > 0)
    min_indices = np.min(data_nonzero, axis=1)
    max_indices = np.max(data_nonzero, axis=1)
    return min_indices, max_indices


def get_crop_transform(min_indices: np.ndarray, max_indices: np.ndarray, shape: tuple) -> tio.Crop:
    crop_sides = []
    for i, (low, high) in enumerate(zip(min_indices, max_indices)):
        for e in (low, shape[i]-1-high):
            crop_sides.append(e)
    return tio.Crop(crop_sides)
    

class EnsureShapeAtLeastTransform(tio.Transform):
    
    def __init__(self, min_shape: tuple):
        super().__init__(parse_input=False)
        self.min_shape = min_shape
        
    def apply_transform(self, nii: nib.Nifti1Image) -> nib.Nifti1Image:
        target_shape = np.max(np.stack([nii.shape, self.min_shape], axis=0), axis=0)
        T = tio.CropOrPad(target_shape=target_shape)
        return T(nii)


def format_target(values: list[str]):
    try:
        return tuple([float(x) for x in values])
    except ValueError:
        return values[0]
    

def maybe_create_out_dir(dir: Path):
    dir.mkdir(exist_ok=True, parents=False)
    (dir / "images").mkdir(exist_ok=True, parents=False)
    (dir / "labels").mkdir(exist_ok=True, parents=False)
    (dir / "splits").mkdir(exist_ok=True, parents=False)


def fix_overlapping_labels(labels: dict):
    
    nii_data_prior_list = []
    for key in multiclass_label_presedence:
        
        if key not in labels.keys():
            continue
        
        nii_current = labels[key]["nii"] 
        nii_data_current = nii_current.get_fdata().astype(np.uint8)
        for nii_data_prior in nii_data_prior_list:
            nii_data_current = nii_data_current & ~nii_data_prior
            
        nii_data_prior_list.append(nii_data_current)
        
        nii_current = nib.Nifti1Image(nii_data_current.astype(np.uint8), nii_current.affine, nii_current.header)
        nii_current.set_data_dtype(np.uint8)
        labels[key]["nii"] = nii_current

    if len(nii_data_prior_list) > 1:
        assert np.max(np.sum(np.stack(nii_data_prior_list), axis=0)) <= 1

    return labels


def save_nii(nii: nib.Nifti1Image, path: Path, dtype: str, is_label: bool):
    assert "uint" in dtype, "dtype shoudl be uint8 or uint16"
    bits = int(dtype.split("uint")[-1])
    max_value = (2 ** bits) - 1
    if not is_label:
        nii = tio.RescaleIntensity(out_min_max=(0, max_value))(nii)
    data = nii.get_fdata()
    assert np.min(data) >= 0 and np.max(data) <= max_value
    data = data.astype(dtype)
    nii = nib.Nifti1Image(data, nii.affine, nii.header)
    nii.set_data_dtype(dtype)
    nib.save(nii, path)


def nib_load(path: Path | str, lazy: bool = True) -> nib.Nifti1Image:
    if lazy:
        return nib.load(path)
    else:
        nii = nib.load(path)
        return nib.Nifti1Image(nii.get_fdata(), nii.affine, nii.header)