import nibabel as nib
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from cvdseg.lightning_modules.evaluation import overlap_metrics, get_lesion_level_tp_fp_fn_vectors
from monai.metrics import compute_average_surface_distance
import json
import argparse

def generate_sphere(radius):
            assert isinstance(radius, int), "Radius must be an integer."
            x = np.arange(-radius, radius+1)
            y = np.arange(-radius, radius+1)
            z = np.arange(-radius, radius+1)
            xx, yy, zz = np.meshgrid(x, y, z)
            sphere = (xx**2 + yy**2 + zz**2) <= radius**2
            return sphere.astype(int)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_subfolder", default="WMH_ISL3")
    parser.add_argument("--models", nargs="+", type=str)
    return parser.parse_args()

def process(args):

    for model in args.models:

        print(f"Calculating metrics for {model}...")
        
        se = generate_sphere(2)

        gt_dir = Path("/home/jessephitidis/BRICIA/MVH_JPhitidis_PhD/projects/WMH_ISL/data/new/labels")
        pred_files = sorted(Path(f"/home/jessephitidis/BRICIA/MVH_JPhitidis_PhD/projects/WMH_ISL/results/{args.results_subfolder}/{model}").glob("*_pred.nii.gz"))

        metrics = defaultdict(lambda: defaultdict(dict))
        for pred_file in tqdm(pred_files, total=len(pred_files)):
            
            # Get ID
            sub_id = pred_file.name.split("_sub-")[0]
            
            # Load data
            pred_soft_all = nib.load(pred_file).get_fdata()
            pred_binary_all = np.argmax(pred_soft_all, axis=0)
            gt_wmh_path = sorted(gt_dir.glob(f"{sub_id}_*_label-WMH_mask.nii.gz"))
            gt_wmh = nib.load(gt_wmh_path[0]).get_fdata() if len(gt_wmh_path) == 1 else None
            gt_isl_path = sorted(gt_dir.glob(f"{sub_id}_*_label-ISLtotal_mask.nii.gz"))
            gt_isl = nib.load(gt_isl_path[0]).get_fdata() if len(gt_isl_path) == 1 else None
            brain = nib.load(sorted(gt_dir.glob(f"{sub_id}_*_label-BRAIN_mask.nii.gz"))[0]).get_fdata()
            icv = np.sum(brain)
            
            for i, (gt, gt_name) in enumerate(zip([gt_wmh, gt_isl], ["WMH", "ISL"])):
                
                pred_soft = pred_soft_all[i+1]
                pred_binary = (pred_binary_all == i+1).astype(int)
                
                if gt is None:
                    
                    metrics[sub_id]["n_gt"][gt_name] = np.nan
                    metrics[sub_id]["n_ltp"][gt_name] = np.nan
                    metrics[sub_id]["n_lfp"][gt_name] =  np.nan
                    metrics[sub_id]["n_lfn"][gt_name] = np.nan
                    metrics[sub_id]["dsc"][gt_name] = np.nan
                    metrics[sub_id]["pre"][gt_name] = np.nan
                    metrics[sub_id]["rec"][gt_name] = np.nan
                    metrics[sub_id]["dsc2"][gt_name] = np.nan
                    metrics[sub_id]["pre2"][gt_name] = np.nan
                    metrics[sub_id]["rec2"][gt_name] = np.nan
                    metrics[sub_id]["ldsc"][gt_name] = np.nan
                    metrics[sub_id]["lpre"][gt_name] = np.nan
                    metrics[sub_id]["lrec"][gt_name] = np.nan
                    metrics[sub_id]["afvd"][gt_name] = np.nan
                    metrics[sub_id]["asd"][gt_name] = np.nan
                    metrics[sub_id]["ap"][gt_name] = np.nan
                    
                    continue
                
                gt_empty = True if np.sum(gt) == 0 else False
                
                # Voxel-wise metrics
                dsc, pre, rec = overlap_metrics(pred_binary, gt)
                
                # Voxel-wise metrics with 2 mm structuring element
                dsc2, pre2, rec2 = overlap_metrics(pred_binary, gt, struct=se)
                
                # Lesion-level metrics
                pred_vec, gt_vec, (n_pred, n_gt) = get_lesion_level_tp_fp_fn_vectors(pred_binary, gt, return_n=True)
                ldsc, lpre, lrec, (ltp, lfp, lfn) = overlap_metrics(pred_vec, gt_vec, return_tp_fp_fn=True)
                
                # Average surface distance
                pred_for_asd = np.moveaxis(np.eye(2)[pred_binary.astype(int)], -1, 0)[None]
                gt_for_asd = np.moveaxis(np.eye(2)[gt.astype(int)], -1, 0)[None]
                asd = compute_average_surface_distance(pred_for_asd, gt_for_asd).item()
                
                # Absolute volume difference as fraction of ICV
                pred_vol = np.sum(pred_binary) 
                gt_vol = np.sum(gt)
                afvd = np.abs((pred_vol - gt_vol)) / icv
                
                # Average precision
                ap = average_precision_score(gt.flatten(), pred_soft.flatten()) if not gt_empty else np.nan
                
                # Fill in metrics
                metrics[sub_id]["n_gt"][gt_name] = float(n_gt)
                metrics[sub_id]["n_ltp"][gt_name] = float(ltp)
                metrics[sub_id]["n_lfp"][gt_name] = float(lfp)
                metrics[sub_id]["n_lfn"][gt_name] = float(lfn)
                metrics[sub_id]["dsc"][gt_name] = dsc
                metrics[sub_id]["pre"][gt_name] = pre
                metrics[sub_id]["rec"][gt_name] = rec
                metrics[sub_id]["dsc2"][gt_name] = dsc2
                metrics[sub_id]["pre2"][gt_name] = pre2
                metrics[sub_id]["rec2"][gt_name] = rec2
                metrics[sub_id]["ldsc"][gt_name] = ldsc
                metrics[sub_id]["lpre"][gt_name] = lpre
                metrics[sub_id]["lrec"][gt_name] = lrec
                metrics[sub_id]["afvd"][gt_name] = afvd
                metrics[sub_id]["asd"][gt_name] = asd
                metrics[sub_id]["ap"][gt_name] = ap
                
        # Calculate summary metrics
        for metric in ["dsc", "pre", "rec", "dsc2", "pre2", "rec2", "ldsc", "lpre", "lrec", "afvd", "asd", "ap"]:
            for gt_name in ["WMH", "ISL"]:
                metrics["summary"][metric][gt_name] = []
                
        lfp_empty_count, lfp_empty_pres_count, empty_count = {"WMH": 0, "ISL": 0}, {"WMH": 0, "ISL": 0}, {"WMH": 0, "ISL": 0}
        for sub_id in metrics:
            
            if sub_id == "summary":
                continue
            
            for metric in ["dsc", "pre", "rec", "dsc2", "pre2", "rec2", "ldsc", "lpre", "lrec", "afvd", "asd", "ap"]:
                for gt_name in ["WMH", "ISL"]:
                    metrics["summary"][metric][gt_name].append(metrics[sub_id][metric][gt_name])
            
            for gt_name in ["WMH", "ISL"]:
                if metrics[sub_id]["n_gt"][gt_name] == 0:
                    empty_count[gt_name] += 1
                    n_lfp = metrics[sub_id]["n_lfp"][gt_name]
                    lfp_empty_count[gt_name] += n_lfp
                    if n_lfp > 0:
                        lfp_empty_pres_count[gt_name] += 1
                        
        for metric in ["dsc", "pre", "rec", "dsc2", "pre2", "rec2", "ldsc", "lpre", "lrec", "afvd", "asd", "ap"]:
            for gt_name in ["WMH", "ISL"]:
                metrics_not_nan_or_inf = [x for x in metrics["summary"][metric][gt_name] if not np.isnan(x) and not np.isinf(x)]
                metrics["summary"][metric][gt_name] = {
                    "mean": np.mean(metrics_not_nan_or_inf), "std": np.std(metrics_not_nan_or_inf), "n": len(metrics_not_nan_or_inf)
                    }
                
        for gt_name in ["WMH", "ISL"]:
            metrics["summary"]["lfp_empty"][gt_name] = {
                "mean": (lfp_empty_count[gt_name] / empty_count[gt_name]) if empty_count[gt_name] > 0 else 0.0,
                "frac_present": (lfp_empty_pres_count[gt_name] / empty_count[gt_name]) if empty_count[gt_name] > 0 else 0.0,
                "n": empty_count[gt_name]
                }
            
        with open(f"/home/jessephitidis/BRICIA/MVH_JPhitidis_PhD/projects/WMH_ISL/results/{args.results_subfolder}/{model}.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
            
if __name__ == "__main__":
    args = get_args()
    process(args)