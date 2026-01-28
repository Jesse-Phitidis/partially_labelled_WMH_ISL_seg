import pytorch_lightning as pl
import torchio as tio
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile
import nibabel as nib
import numpy as np
import pandas as pd

class Base(pl.LightningDataModule):
    
    def __init__(
        self,
        data_dir: str,
        train_split_path: str | None,
        val_split_path: str | None,
        test_split_path: str | None,
        images: list[str],
        labels: list[str],
        transforms_train: tio.Transform | None,
        transforms_test: tio.Transform | None,
        sampler: tio.data.PatchSampler | None,
        queue_max_length: int = 400,
        queue_samples_per_volume: int = 1,
        queue_num_workers: int = 8,
        batch_size: int = 1,
        val_as_test: bool = False,
        missing_images_ok: bool = False,
        missing_labels_ok: bool = False,
        use_pseudolabels: bool = False,
        predict_split_path: str | None = None
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train_split = pd.read_csv(train_split_path, header=None).values.squeeze().tolist() if train_split_path is not None else None
        self.val_split = pd.read_csv(val_split_path, header=None).values.squeeze().tolist() if val_split_path is not None else None
        self.test_split = pd.read_csv(test_split_path, header=None).values.squeeze().tolist() if test_split_path is not None else None
        self.images = images
        self.labels = labels
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test 
        self.sampler = sampler
        self.queue_max_length = queue_max_length
        self.queue_samples_per_volume = queue_samples_per_volume
        self.queue_num_workers = queue_num_workers
        self.batch_size = batch_size
        self.val_as_test = val_as_test
        self.missing_images_ok = missing_images_ok
        self.missing_labels_ok = missing_labels_ok
        self.use_pseudolabels = use_pseudolabels
        self.predict_split = pd.read_csv(predict_split_path, header=None).values.squeeze().tolist() if predict_split_path is not None else None
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def build_dataset(self, ids: list, transforms: tio.Transform, sampling_label: bool = False, use_pseudolabels: bool = False) -> tio.SubjectsDataset:
        subjects = []
        for id in ids:
            missing_images, missing_labels = self.get_missing(id=id)
            for lst in [missing_images, missing_labels]:
                if len(lst) != 0:
                    print(f"Subject id: {id} is missing {lst}")
            if len(missing_images) == len(self.images):
                print(f"Subject id: {id} has no images, skipping")
                continue
            if len(missing_labels) == len(self.labels):
                print(f"Subject id: {id} has no labels, skipping")
                continue
            subject_dict = {}   
            if (len(missing_images) == 0 or self.missing_images_ok) and (len(missing_labels) == 0 or self.missing_labels_ok):
                for sequence in self.images:
                    if sequence not in missing_images:
                        file = list((self.data_dir / "images").glob(f"{id}_*{sequence}*"))[0]
                        subject_dict[sequence] = tio.ScalarImage(file, reader=tio.data.io._read_nibabel)
                    else:
                        file = list((self.data_dir / "images").glob(f"{id}_*"))[0]
                        ref = nib.load(file)
                        nans = torch.ones(ref.shape).unsqueeze(0) * torch.nan
                        subject_dict[sequence] = tio.ScalarImage(tensor=nans, reader=tio.data.io._read_nibabel, affine=ref.affine)
                for label in self.labels:
                    if label not in missing_labels:
                        file = list((self.data_dir / "labels").glob(f"{id}_*-{label}_*"))[0]
                        subject_dict[label] = tio.LabelMap(file, reader=tio.data.io._read_nibabel)
                    else:
                        if use_pseudolabels:
                            file = list((self.data_dir / "labels").glob(f"{id}_*-pseudo{label}_*"))[0]
                            subject_dict[label] = tio.LabelMap(file, reader=tio.data.io._read_nibabel)
                            print(f"Using pseudolabel for {label}")
                        else:
                            file = list((self.data_dir / "labels").glob(f"{id}_*"))[0]
                            ref = nib.load(file)
                            nans = torch.ones(ref.shape).unsqueeze(0) * torch.nan
                            subject_dict[label] = tio.LabelMap(tensor=nans, reader=tio.data.io._read_nibabel, affine=ref.affine)
                if sampling_label: # So we can sample patches using torchio's LabelSampler with multilabel setting (only used for sampling)
                    self.add_sampling_label(subject_dict, id)
                subjects.append(tio.Subject(subject_dict))
        return tio.SubjectsDataset(subjects=subjects, transform=transforms)
    
    def get_missing(self, id: str) -> tuple:
        missing_images, missing_labels = [], []
        for sequence in self.images:
            file = list((self.data_dir / "images").glob(f"{id}*{sequence}*"))
            if len(file) == 0:
                missing_images.append(sequence)
        for label in self.labels:
            file = list((self.data_dir / "labels").glob(f"{id}_*-{label}_*"))
            if len(file) == 0:
                missing_labels.append(label)
        return missing_images, missing_labels
    
    def add_sampling_label(self, subject_dict: dict, id: str) -> None:
    
        for v in subject_dict.values():
            if isinstance(v, tio.LabelMap):
                path = v["path"]
                if Path(path).is_file():
                    ref = nib.load(path)
                    affine, header = ref.affine, ref.header
                    break
        
        data = torch.cat([subject_dict[label]["data"] for label in self.labels if label != "ANAT"], dim=0)
        data[torch.isnan(data)] = 0 # might need this 
        bg = 1 - torch.max(data, dim=0, keepdim=True).values
        data = torch.cat([bg, data], dim=0)
        im = nib.Nifti1Image(data.numpy().astype(np.uint8), affine, header)
        im.set_data_dtype(np.uint8)
        save_path = Path(self.temp_dir.name) / (id + "_sampling_label.nii.gz")
        nib.save(im, save_path)
        subject_dict["sampling_label"] = tio.LabelMap(save_path, reader=tio.data.io._read_nibabel)
        
    def setup(self, stage: str) -> None:
        if stage.lower() == "fit":
            sampling_label=True if self.sampler.probability_map_name == "sampling_label" else False
            self.dataset_train = self.build_dataset(
                ids=self.train_split, transforms=self.transforms_train, sampling_label=sampling_label, use_pseudolabels=self.use_pseudolabels
                )
            self.dataset_val = self.build_dataset(ids=self.val_split, transforms=self.transforms_test)
        if stage.lower() == "test":
            self.dataset_test = self.build_dataset(ids=self.test_split if not self.val_as_test else self.val_split, transforms=self.transforms_test)
        if stage.lower() == "predict":
            print("WARNING: predict step used for generating pseudolabels, not general inference")
            self.dataset_predict = self.build_dataset(ids=self.predict_split, transforms=self.transforms_test)
            
    def train_dataloader(self) -> DataLoader:
        queue = tio.data.Queue(
            subjects_dataset=self.dataset_train,
            max_length=self.queue_max_length,
            samples_per_volume=self.queue_samples_per_volume,
            sampler=self.sampler,
            num_workers=self.queue_num_workers
            )
        return DataLoader(dataset=queue, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset_val, batch_size=1,shuffle=False, num_workers=0)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset_test, batch_size=1,shuffle=False, num_workers=0)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset_predict, batch_size=1,shuffle=False, num_workers=0)
    
    def val_dataloader_temp_scaling(self, patch_size: np.ndarray, images: list, labels: list) -> DataLoader:
        all_images, all_labels = self.images, self.labels
        self.images, self.labels = images, labels
        dataset_val = self.build_dataset(ids=self.val_split, transforms=self.transforms_test)
        self.images, self.labels = all_images, all_labels
        self.sampler.patch_size = patch_size
        queue = tio.data.Queue(
            subjects_dataset=dataset_val,
            max_length=self.queue_max_length,
            samples_per_volume=self.queue_samples_per_volume,
            sampler=self.sampler,
            num_workers=self.queue_num_workers
            )
        return DataLoader(dataset=queue, batch_size=self.batch_size, shuffle=False, num_workers=0)