import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryCalibrationError
from typing import Any, Literal
from omegaconf import OmegaConf
from pathlib import Path
from cvdseg.lightning_modules import utils
from collections import defaultdict
import nibabel as nib
import numpy as np
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from cvdseg.lightning_modules.evaluation import NonMONAIMetrics
from cvdseg.lightning_modules.temp_scaling import find_temp
import json
from cvdseg.lightning_modules import constants as C
import sys

SurfaceDistanceMetric.aggregate.__globals__["do_metric_reduction"] = utils.my_do_metric_reduction
DiceMetric.__init__.__globals__["DiceHelper"].compute_channel = utils.my_compute_channel

class Ensemble(pl.LightningModule):

    def __init__(
            self, 
            seedless_log_paths: list, 
            activation: Literal["softmax", "sigmoid", "both"],
            test_post_processor: Any, 
            seeds: list | None = None, 
            config: dict | None = None, 
            ckpt_type: Literal["best", "final"] = "best",
            network_kwargs: list | None = None,
            test_pred_dir: str | None = None,
            predict_pred_dir: str | None = None,
            test_metrics_path: str | None = None,
            calculate_ece: bool = False,
            temp_scaling: Literal["auto"] | list[list] | None = None
            ):
        
        super().__init__()
        self.activation = activation
        self.test_post_processor = test_post_processor
        self.seedless_log_paths = seedless_log_paths
        self.seeds = [f"_{s}" for s in seeds] if seeds is not None else [""]
        self.ckpt_type = ckpt_type
        self.network_kwargs = network_kwargs if network_kwargs is not None else [{}] * len(seedless_log_paths) # Maybe could require different for different models? Oh well
        self.test_pred_dir = Path(test_pred_dir) if test_pred_dir is not None else None
        self.predict_pred_dir = Path(predict_pred_dir) if predict_pred_dir is not None else None
        self.test_metrics_path = Path(test_metrics_path) if test_metrics_path is not None else None
        self.calculate_ece = calculate_ece
        self.temp_scaling = temp_scaling
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)
        self.surface_distance_metric = SurfaceDistanceMetric(include_background=True, reduction="mean_batch")
        self.non_monai_metrics = NonMONAIMetrics(include_background=True, reduction="mean")
        self.get_networks(config if config is not None else {})

    def setup(self, stage: str) -> None:
        if stage not in ["test", "predict"]:
            raise ValueError("Ensemble module is only for test or predict stage")

    def get_networks(self, config: dict):
        self.input_images_lst = []
        self.output_labels_lst = []
        self.inferers_lst = []
        self.ensemble = nn.ModuleList()
        for i, seed in enumerate(self.seeds):
            networks = nn.ModuleList()
            for seedless_log_path in self.seedless_log_paths:
                network, input_images, output_labels, inferer = self.instantiate_network_and_get_settings(seedless_log_path, seed, config)
                networks.append(network)
                self.input_images_lst.append(input_images) if i==0 else None
                self.output_labels_lst.append(output_labels) if i==0 else None
                self.inferers_lst.append(inferer) if i==0 else None
            self.ensemble.append(networks)
    
    def set_temp_scaling(self) -> None:
        if self.temp_scaling is None:
            # self.temp_scaling = [[1] * len(seeds) for models in range(len(self.ensemble)) for seeds in range(len(models))]
            self.temp_scaling = [[1] * len(self.seedless_log_paths) for _ in range(len(self.seeds))]
        if self.temp_scaling == "auto":
            self.temp_scaling = self.do_temp_scaling()
            config_path = Path(sys.argv[sys.argv.index("--config")+1])
            config = OmegaConf.load(config_path)
            config.model.init_args.temp_scaling = self.temp_scaling
            OmegaConf.save(config, config_path) 
            
    def do_temp_scaling(self) -> list[list]:
        temps_ensemble = []
        for networks, seed in zip(self.ensemble, self.seeds):
            temps_network = []
            for i, (network, images, labels, inferer, seedless_log_path) in enumerate(zip(networks, self.input_images_lst, self.output_labels_lst, self.inferers_lst, self.seedless_log_paths)):
                patch_size = np.array(inferer.roi_size)
                loader = self.trainer.datamodule.val_dataloader_temp_scaling(patch_size, images, labels)
                temp, ori_nll, new_nll = find_temp(network, loader, images, labels, **self.network_kwargs[i])
                temps_network.append(temp)
                print(f"{seedless_log_path + seed}:   temp = {temp:.3f},   nll = {ori_nll:.5f} --> {new_nll:.5f}")
            temps_ensemble.append(temps_network)
        return temps_ensemble
                
    def do_activation(self, x: torch.Tensor, labels: list) -> torch.Tensor:
        if self.activation == "softmax":
            return torch.softmax(x, dim=1)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "both":
            n = 0
            for lab in labels:
                n += 1 if lab in C.mutually_exclusive_labels else None
            mutually_exclusive_channels = x[:, :n+1, ...]
            independent_channels = x[:, n+1:, ...]
            return torch.cat([torch.softmax(mutually_exclusive_channels, dim=1), torch.sigmoid(independent_channels)], dim=1)
    
    def instantiate_network_and_get_settings(self, seedless_log_path: str, seed: int, config: dict) -> tuple[nn.Module, list]:
        full_log_path = Path(seedless_log_path + seed)
        base_config = OmegaConf.load(full_log_path / "config.yaml")
        merged_config = OmegaConf.merge(base_config, config)
        ckpt_name = "epoch=*-dice=*.ckpt" if self.ckpt_type == "best" else "epoch=*.ckpt"
        ckpt = list(full_log_path.rglob(ckpt_name))[0]
        network = utils.instantiate_object_from_config(OmegaConf.to_container(merged_config.model.init_args.network))
        state_dict = {utils.split_join(k, ".", slice(1, None)): v for k, v in torch.load(ckpt)["state_dict"].items() if k.startswith("network")}
        network.load_state_dict(state_dict)
        images = merged_config.data.init_args.images
        labels = merged_config.data.init_args.labels
        inferer = utils.instantiate_object_from_config(OmegaConf.to_container(merged_config.model.init_args.test_inferer))
        return network, images, labels, inferer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensemble_preds = []
        for networks, temp_scaling_lst in zip(self.ensemble, self.temp_scaling):
            network_preds = []
            for i, (network, images, labels, inferer, temp) in enumerate(zip(networks, self.input_images_lst, self.output_labels_lst, self.inferers_lst, temp_scaling_lst)):
                inp = self.extract_required_input_channels(x, images)
                pred = inferer(inp, network, **self.network_kwargs[i]) / temp
                network_preds.append(pred)
            ensemble_preds.append(self.do_activation(torch.cat(network_preds, dim=1), labels))
        return torch.mean(torch.stack(ensemble_preds, dim=0), dim=0)

    def extract_required_input_channels(self, x: torch.Tensor, images: list) -> torch.Tensor:
        indices = []
        for image in images:
            indices.append(self.trainer.datamodule.images.index(image))
        return x[:, indices, ...]

    def on_test_start(self) -> None:
        assert self.trainer.datamodule.batch_size == 1,  "only batch size of one currently implemented for metric calculation"
        if self.test_pred_dir:
            self.test_pred_dir.mkdir(exist_ok=True, parents=False)
        if self.test_metrics_path:
            self.test_metrics_path.parent.mkdir(exist_ok=True, parents=False)
            self.test_metrics_json = {}
        if self.calculate_ece:
            self.ece_metrics = [BinaryCalibrationError() for _ in range(len(self.trainer.datamodule.labels))]
        self.set_temp_scaling()

    def test_step(self, batch: dict) -> None:
        images, labels = self.extract_data_from_batch(batch)
        pred_soft = self(images)
        pred = self.test_post_processor(self, pred_soft)
        pred = self.get_only_requested_preds(pred)
        dice = self.dice_metric(pred, labels)
        surface_distance = self.surface_distance_metric(pred, labels)
        pre, rec, lf1, lpre, lrec = self.non_monai_metrics(pred, labels)
        if self.calculate_ece:
            ece = [metric(pred_soft[:,i,...].cpu(), labels[:,i,...].cpu()) for i, metric in enumerate(self.ece_metrics)]
            ece = torch.stack(ece)

        test_dict = defaultdict(dict)
        for i, lab in enumerate(self.trainer.datamodule.labels):
            test_dict["dice"].update({lab: dice[0, i].item()})
            test_dict["surface_distance"].update({lab: surface_distance[0, i].item()})
            test_dict["precision"].update({lab: pre[0, i].item()})
            test_dict["recall"].update({lab: rec[0, i].item()})
            test_dict["lesion_f1"].update({lab: lf1[0, i].item()})
            test_dict["lesion_precision"].update({lab: lpre[0, i].item()})
            test_dict["lesion_recall"].update({lab: lrec[0, i].item()})
            if self.calculate_ece:
                test_dict["ece"].update({lab: ece[i].item()})
        test_dict["dice"].update({"mean": utils.safe_mean(dice).item()})
        test_dict["surface_distance"].update({"mean": utils.safe_mean(surface_distance).item()})
        test_dict["precision"].update({"mean": utils.safe_mean(pre).item()})
        test_dict["recall"].update({"mean": utils.safe_mean(rec).item()})
        test_dict["lesion_f1"].update({"mean": utils.safe_mean(lf1).item()})
        test_dict["lesion_precision"].update({"mean": utils.safe_mean(lpre).item()})
        test_dict["lesion_recall"].update({"mean": utils.safe_mean(lrec).item()})
        if self.calculate_ece:
            test_dict["ece"].update({"mean": utils.safe_mean(ece).item()})

        if self.test_pred_dir:
            self.save_pred(pred, batch)
        if self.test_metrics_path:
            self.add_test_metrics(
                test_dict, final=False, batch=batch
            )
            
    def get_only_requested_preds(self, pred: torch.Tensor) -> torch.Tensor:
        indices = []
        for lab in self.trainer.datamodule.labels:
            indices.append([l for lst in self.output_labels_lst for l in lst].index(lab))
        return pred[:, indices, ...]

    def on_test_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate()
        surface_distance = self.surface_distance_metric.aggregate()
        pre, rec, lf1, lpre, lrec = self.non_monai_metrics.aggregate()
        if self.calculate_ece:
            ece = [metric.compute() for metric in self.ece_metrics]
            ece = torch.stack(ece)
        self.dice_metric.reset()
        self.surface_distance_metric.reset()
        self.non_monai_metrics.reset()

        test_dict = defaultdict(dict)
        for i, lab in enumerate(self.trainer.datamodule.labels):
            test_dict["dice"].update({lab: dice[i].item()})
            test_dict["surface_distance"].update({lab: surface_distance[i].item()})
            test_dict["precision"].update({lab: pre[i].item()})
            test_dict["recall"].update({lab: rec[i].item()})
            test_dict["lesion_f1"].update({lab: lf1[i].item()})
            test_dict["lesion_precision"].update({lab: lpre[i].item()})
            test_dict["lesion_recall"].update({lab: lrec[i].item()})
            if self.calculate_ece:
                test_dict["ece"].update({lab: ece[i].item()})
        test_dict["dice"].update({"mean": utils.safe_mean(dice).item()})
        test_dict["surface_distance"].update({"mean": utils.safe_mean(surface_distance).item()})
        test_dict["precision"].update({"mean": utils.safe_mean(pre).item()})
        test_dict["recall"].update({"mean": utils.safe_mean(rec).item()})
        test_dict["lesion_f1"].update({"mean": utils.safe_mean(lf1).item()})
        test_dict["lesion_precision"].update({"mean": utils.safe_mean(lpre).item()})
        test_dict["lesion_recall"].update({"mean": utils.safe_mean(lrec).item()})
        if self.calculate_ece:
            test_dict["ece"].update({"mean": utils.safe_mean(ece).item()})

        if self.test_metrics_path:
            self.add_test_metrics(test_dict, final=True)
            with open(self.test_metrics_path, "w") as f:
                json.dump(self.test_metrics_json, f, indent=4)
        config_file = Path.cwd() / "config.yaml"
        if config_file.exists():
            config_file.unlink()
            print("Deleted test config")

        for metric, result in test_dict.items():
            print(f"\n{metric}:")
            for lab, score in result.items():
                print(f"    {lab.ljust(20)}: {score:.4f}")
        print()
        
    def on_predict_start(self) -> None:
        assert self.trainer.datamodule.batch_size == 1,  "only batch size of one currently implemented for metric calculation"
        if self.predict_pred_dir:
            self.predict_pred_dir.mkdir(exist_ok=True, parents=False)
        self.set_temp_scaling()
        
    def predict_step(self, batch: dict) -> None:
        images, labels = self.extract_data_from_batch(batch)
        pred_soft = self(images)
        pred = self.test_post_processor(self, pred_soft)

        # Correct as much as possible the predictions for pseudolabels based on the labels we do have already
        n_channels = labels.shape[1]
        channels_for_predicting = []
        for c in range(n_channels):
            lab_c = labels[:, c:c+1]
            if torch.any(torch.isnan(lab_c)):
                channels_for_predicting.append(c)
            else:
                for i in range(n_channels):
                    if i != c:
                        pred[:, i:i+1][lab_c==1] = 0
                        
        if self.predict_pred_dir:
            self.save_pseudo_pred(pred, batch, channels_for_predicting)

    def save_pred(self, pred: torch.Tensor, batch: dict) -> None:
        key = self.trainer.datamodule.images[0]
        save_name_generic = utils.split_join(str(batch[key]["stem"][0]), "_", slice(None, -1)) + "_%s_pred.nii.gz"
        original_path = batch[key]["path"][0]
        original_image = nib.load(original_path)
        header, affine = original_image.header, original_image.affine
        pred = pred.cpu().numpy().squeeze(0)
        for i, label in enumerate(self.trainer.datamodule.labels):
            channel_pred = pred[i, ...]
            save_name = save_name_generic % label
            save_path = self.test_pred_dir / save_name
            im = nib.Nifti1Image(channel_pred.astype("uint8"), affine, header)
            im.set_data_dtype("uint8")
            nib.save(im, save_path)
              
    def save_pseudo_pred(self, pred: torch.Tensor, batch: dict, channels_for_predicting: list) -> None:
        key = self.trainer.datamodule.images[0]
        save_name_generic = utils.split_join(str(batch[key]["stem"][0]), "_", slice(None, -1)) + "_label-%s_mask.nii.gz"
        original_path = batch[key]["path"][0]
        original_image = nib.load(original_path)
        header, affine = original_image.header, original_image.affine
        pred = pred.cpu().numpy().squeeze(0)
        for i, label in enumerate(self.trainer.datamodule.labels):
            if i not in channels_for_predicting:
                continue
            channel_pred = pred[i, ...]
            save_name = save_name_generic % ("pseudo" + label)
            save_path = self.predict_pred_dir / save_name
            im = nib.Nifti1Image(channel_pred.astype("uint8"), affine, header)
            im.set_data_dtype("uint8")
            nib.save(im, save_path)

    def add_test_metrics(self, metrics: dict, final: bool, batch: dict = None) -> None:
        if not final:
            assert batch, "Batch must be provided when adding non-final test metrics"
            self.test_metrics_json[utils.split_join(batch[self.trainer.datamodule.images[0]]["stem"][0], "_", slice(None, -1))] = metrics
        else:
            self.test_metrics_json["mean"] = metrics
    
    def extract_data_from_batch(self, batch: dict) -> tuple[torch.Tensor]:
        images_lst = []
        for image in self.trainer.datamodule.images:
            images_lst.append(batch[image]["data"])
        images = torch.cat(images_lst, dim=1)
        
        labels_lst = []
        for label in self.trainer.datamodule.labels:
            labels_lst.append(batch[label]["data"])
        labels = torch.cat(labels_lst, dim=1)

        return images, labels
        
    


