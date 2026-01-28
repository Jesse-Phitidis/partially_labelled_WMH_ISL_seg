import pytorch_lightning as pl
import torch
import torch.nn as nn
import monai
from cvdseg.lightning_modules import losses
import numpy as np
from pathlib import Path
from cvdseg.lightning_modules import utils
import nibabel as nib
from collections import defaultdict
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from typing import Any, Callable
from cvdseg.lightning_modules.losses import MyDeepSupervisionLoss, MyDeepSupervisionLoss_DefaultConfig
import json

SurfaceDistanceMetric.aggregate.__globals__["do_metric_reduction"] = utils.my_do_metric_reduction
DiceMetric.__init__.__globals__["DiceHelper"].compute_channel = utils.my_compute_channel

class Base(pl.LightningModule):
    
    def __init__(
        self,
        network: nn.Module,
        loss_fn: nn.Module | dict,
        test_inferer: monai.inferers.Inferer,
        test_post_processor: Any,
        network_kwargs: dict = {},
        deep_supr_loss_config: dict = MyDeepSupervisionLoss_DefaultConfig,
        watch_log_freq: int = 100,
        surface_distance_freq: int = 5, # not used
        test_pred_dir: str | None = None,
        test_metrics_path: str | None = None,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(ignore=["network"])
        self.network = network
        self.criterion =  MyDeepSupervisionLoss(loss=self.get_loss_fn(loss_fn), **deep_supr_loss_config)
        self.test_inferer = test_inferer
        self.test_post_processor = test_post_processor
        self.network_kwargs = network_kwargs
        self.watch_log_freq = watch_log_freq
        # self.surface_distance_freq = surface_distance_freq
        self.test_pred_dir = Path(test_pred_dir) if test_pred_dir else None
        self.test_metrics_path = Path(test_metrics_path) if test_metrics_path else None
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)
        # self.surface_distance_metric = SurfaceDistanceMetric(include_background=True, reduction="mean_batch")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.network(x, **self.network_kwargs)
        spatial_dims = len(x.shape[2:])
        if len(pred.shape) == 2 + spatial_dims:
            return pred
        else:
            return torch.unbind(pred, dim=1)
    
    def on_train_start(self) -> None:
        self.logger.watch(self.network, log="all", log_freq=self.watch_log_freq)
        
    def training_step(self, batch: dict) -> torch.Tensor:
        images, labels = self.extract_data_from_batch(batch, train=True)
        pred = self(images)
        loss = self.criterion(pred, labels)
        self.log("loss", loss.item(), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: dict) -> None:
        images, labels = self.extract_data_from_batch(batch, train=False)
        pred = self.test_post_processor(self, self.test_inferer(images, self))
        self.dice_metric(pred, labels)
        # if (self.current_epoch + 1) % (self.surface_distance_freq * self.trainer.check_val_every_n_epoch) == 0:
        #     self.surface_distance_metric(pred, labels)
        return pred

    def on_validation_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate()
        self.dice_metric.reset()
        val_dict = {}
        for i, lab in enumerate(self.trainer.datamodule.labels):
            val_dict[f"dice/{lab}"] = dice[i].item()
        val_dict["dice/mean"] = utils.safe_mean(dice).item()

        # if (self.current_epoch + 1) % (self.surface_distance_freq * self.trainer.check_val_every_n_epoch) == 0:
        #     surface_distance = self.surface_distance_metric.aggregate()
        #     self.surface_distance_metric.reset()
        #     for i, lab in enumerate(self.trainer.datamodule.labels):
        #         val_dict[f"surface_distance/{lab}"] = surface_distance[i].item()
        #     val_dict["surface_distance/mean"] = surface_distance.mean().item()

        self.log_dict(val_dict, on_step=False, on_epoch=True)

    def on_test_start(self) -> None:
        assert self.trainer.datamodule.batch_size == 1,  "only batch size of one currently implemented for metric calculation"
        if self.test_pred_dir:
            self.test_pred_dir.mkdir(exist_ok=True, parents=False)
        if self.test_metrics_path:
            self.test_metrics_path.parent.mkdir(exist_ok=True, parents=False)
            self.test_metrics_json = {}

    def test_step(self, batch: dict) -> None:
        images, labels = self.extract_data_from_batch(batch, train=False)
        pred = self.test_post_processor(self, self.test_inferer(images, self))
        dice = self.dice_metric(pred, labels)
        # surface_distance = self.surface_distance_metric(pred, labels)

        test_dict = defaultdict(dict)
        for i, lab in enumerate(self.trainer.datamodule.labels):
            test_dict["dice"].update({lab: dice[0, i].item()})
            # test_dict["surface_distance"].update({lab: surface_distance[0, i].item()})
        test_dict["dice"].update({"mean": utils.safe_mean(dice).item()})
        # test_dict["surface_distance"].update({"mean": utils.safe_mean(surface_distance).item()})

        if self.test_pred_dir:
            self.save_pred(pred, batch)
        if self.test_metrics_path:
            self.add_test_metrics(
                test_dict, final=False, batch=batch
            )

    def on_test_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate()
        # surface_distance = self.surface_distance_metric.aggregate()
        self.dice_metric.reset()
        # self.surface_distance_metric.reset()

        test_dict = defaultdict(dict)
        for i, lab in enumerate(self.trainer.datamodule.labels):
            test_dict["dice"].update({lab: dice[i].item()})
            # test_dict["surface_distance"].update({lab: surface_distance[i].item()})
        test_dict["dice"].update({"mean": utils.safe_mean(dice).item()})
        # test_dict["surface_distance"].update({"mean": utils.safe_mean(surface_distance).item()})
        
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
                print(f"    {lab.ljust(10)}: {score:.4f}")
        print()
        
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

    def add_test_metrics(self, metrics: dict, final: bool, batch: dict = None) -> None:
        if not final:
            assert batch, "Batch must be provided when adding non-final test metrics"
            self.test_metrics_json[utils.split_join(batch[self.trainer.datamodule.images[0]]["stem"][0], "_", slice(None, -1))] = metrics
        else:
            self.test_metrics_json["mean"] = metrics
    
    def extract_data_from_batch(self, batch: dict, train: bool) -> tuple[torch.Tensor]:
        raise NotImplementedError
    
    def get_loss_fn(self, loss_fn: nn.Module | dict) -> nn.Module:
        raise NotImplementedError
