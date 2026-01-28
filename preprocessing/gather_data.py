'''
Gathers data from MVH_JPhitidis_PhD/data/datasets for specific sheet in sheets dir. Preprocesses and saves with name "dataset_id_" + filepath.name.
Also saves a csv file with a list of "dataset_id" e.g. main_test.csv.
'''
import argparse
import logging
import pandas as pd
from pathlib import Path 
import nibabel as nib
import pputils as pp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--datasets_loc", type=str, required=True, help="Path to the directory containing all datasets in BIDS format")
    parser.add_argument("--out_dir", type=str, required=True) # e.g. data/main
    parser.add_argument("--images", type=str, required=True, nargs="+")
    parser.add_argument("--labels", type=str, required=True, nargs="+")
    parser.add_argument("--pvs_src", type=str, required=False, default=["T2W", "all"], nargs="+", help="Sequence for which GT PVS derived")
    parser.add_argument("--target", type=str, required=True, nargs="+", help="Target spacing or image")
    parser.add_argument("--image_interp", type=str, default="linear", choices=["linear", "bspline", "nearest"])
    parser.add_argument("--label_interp", type=str, default="linear", choices=["linear", "nearest"])
    parser.add_argument("--min_shape", type=float, required=False, default=None, nargs="+", help="Min spatial shape of processed data")
    parser.add_argument("--image_dtype", type=str, default="uint8", choices=["uint8", "uint16"])
    parser.add_argument("--label_dtype", type=str, default="uint8", choices=["uint8", "uint16"])
    return parser.parse_args()


def get_logger():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def process_dataset(args, logger):

    # Get paths as Path objects
    csv = Path(args.csv)
    datasets_loc = Path(args.datasets_loc)
    out_dir = Path(args.out_dir)
    # Load the csv as a pandas DataFrame
    df = pd.read_csv(csv, converters={"id": str})
    df.fillna("N/A", inplace=True)
    # Create the output directories if required
    pp.maybe_create_out_dir(out_dir)
    # List to store dataset_id in a csv file
    csv_out_list = []

    for _, subject in tqdm(df.iterrows(), total=len(df)):

        csv_out_list.append(f'{subject["dataset"]}_{subject["id"]}')
        
        subject_already_done = False
        
        images, labels = {}, {}
        for key in args.images + args.labels:
            
            subdir = "images" if key in args.images else "labels"
            file_out_path = out_dir / subdir / (f'{subject["dataset"]}_{subject["id"]}_{subject[key].split("/")[-1]}')
            if file_out_path.exists():
                logger.info(f"{subject['id']}:   already processed")
                subject_already_done = True
                break # This assumes that we always preprocess the same images and labels. If this is not true then
                      # change to continue, but be aware that it may cause different image shapes for the same subject 
            
            if key=="PVS" and subject["PVSsrc"].lower() not in [x.lower() for x in args.pvs_src]:
                logger.warning(f"{subject['id']}:   PVSsrc is {subject['PVSsrc']} but using {args.pvs_src}. Skipping PVS")
                continue
            
            nii_path = datasets_loc / subject["dataset"] / subject[key]
            try:
                nii = pp.nib_load(nii_path)
            except FileNotFoundError:
                logger.warning(f"{subject['id']}:   {key} not found")
                continue
            
            rel_dict = images if key in args.images else labels
            rel_dict[key] = {"nii": nii, "out_path": file_out_path}
        
        if subject_already_done:
            continue
        
        assert len(images) != 0 and len(labels) != 0, f"{subject['id']} has no images or labels but not already done..."
        
        images, labels = pp.process_subject(
            images=images,
            labels=labels,
            target=args.target,
            image_interp=args.image_interp,
            label_interp=args.label_interp,
            min_shape=args.min_shape,
        )

        for k, v in images.items():
            if not v["out_path"].exists():
                pp.save_nii(nii=v["nii"], path=v["out_path"], dtype=args.image_dtype, is_label=False)

        labels = pp.fix_overlapping_labels(labels=labels)
        for k, v in labels.items():
            if not v["out_path"].exists():
                pp.save_nii(nii=v["nii"], path=v["out_path"], dtype=args.label_dtype, is_label=True)

    csv_out_series = pd.Series(csv_out_list)
    csv_out_series.to_csv(out_dir / "splits" / csv.name, index=False, header=False)


def main():
    args = parse_args()
    args.target = pp.format_target(args.target)
    logger = get_logger()
    process_dataset(args, logger)

if __name__ == "__main__":
    main()