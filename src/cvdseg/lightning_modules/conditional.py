from cvdseg.lightning_modules.base import Base
from cvdseg.lightning_modules.multi_class import Multiclass
from cvdseg.lightning_modules.ensemble import Ensemble

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

class ConditionalMulticlass(Multiclass):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, condition = x[:, :-1], x[:, -1:]
        pred = self.network(x, condition, **self.network_kwargs)
        spatial_dims = len(x.shape[2:])
        if len(pred.shape) == 2 + spatial_dims:
            return pred
        else:
            return torch.unbind(pred, dim=1)
        
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
    
    def on_test_start(self) -> None:
        raise NotImplementedError
    
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
            
    def extract_data_from_batch(self, batch: dict, train: bool) -> tuple[torch.Tensor]:
        
        images_lst = []
        for image in self.trainer.datamodule.images:
            images_lst.append(batch[image]["data"])
        images = torch.cat(images_lst, dim=1)
        
        labels_lst = []
        for label in self.trainer.datamodule.labels:
            labels_lst.append(batch[label]["data"])
        
        if train:
            bg = 1 - torch.max(torch.stack(labels_lst, dim=0), dim=0).values
            labels_lst.insert(0, bg)
            
        labels = torch.cat(labels_lst, dim=1)
        
        condition = batch[self.trainer.datamodule.conditional_label]["data"]
        
        images = torch.cat([images, condition], dim=1)

        return images, labels
    
    
    
class ConditionalMulticlassEnsemble(Ensemble):
    
    def get_callable(network: nn.Module, **network_kwargs: dict):
        def callable(x: torch.tensor):
            x, condition = x[:, :-1], x[:, -1:]
            pred = network(x, condition, **network_kwargs)
            return pred
        return callable
    
    def forward(self, x: torch.tensor) -> torch.Tensor:
        x, condition = x[:,:-1], x[:,-1:]
        ensemble_preds = []
        for networks, temp_scaling_lst in zip(self.ensemble, self.temp_scaling):
            network_preds = []
            for i, (network, images, labels, inferer, temp) in enumerate(zip(networks, self.input_images_lst, self.output_labels_lst, self.inferers_lst, temp_scaling_lst)):
                inp = self.extract_required_input_channels(x, images)
                pred = inferer(torch.cat([inp, condition], dim=1), self.get_callable(network, **self.network_kwargs[i])) / temp
                network_preds.append(pred)
            ensemble_preds.append(self.do_activation(torch.cat(network_preds, dim=1), labels))
        return torch.mean(torch.stack(ensemble_preds, dim=0), dim=0)
    
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
    
    def extract_data_from_batch(self, batch: dict) -> tuple[torch.Tensor]:
        images_lst = []
        for image in self.trainer.datamodule.images:
            images_lst.append(batch[image]["data"])
        images = torch.cat(images_lst, dim=1)
        
        labels_lst = []
        for label in self.trainer.datamodule.labels:
            labels_lst.append(batch[label]["data"])
        labels = torch.cat(labels_lst, dim=1)
        
        condition = batch[self.trainer.datamodule.conditional_label]["data"]
        
        images = torch.cat([images, condition], dim=1)

        return images, labels