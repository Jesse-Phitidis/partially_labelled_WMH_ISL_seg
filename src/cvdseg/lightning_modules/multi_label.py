from cvdseg.lightning_modules.base import Base
import torch
import torch.nn as nn
from typing import Sequence

class Multilabel(Base):
        
    def get_loss_fn(self, loss_fn: nn.Module) -> nn.Module:
        return loss_fn
        
    def extract_data_from_batch(self, batch: dict, train: bool) -> tuple[torch.Tensor]:
        
        images_lst = []
        for image in self.trainer.datamodule.images:
            images_lst.append(batch[image]["data"])
        images = torch.cat(images_lst, dim=1)
        
        labels_lst = []
        for label in self.trainer.datamodule.labels:
            labels_lst.append(batch[label]["data"])
        labels = torch.cat(labels_lst, dim=1)

        return images, labels
    
    def on_test_start(self) -> None:
        raise NotImplementedError
    

class MultilabelWithANATInput(Base):
        
    def get_loss_fn(self, loss_fn: nn.Module) -> nn.Module:
        return loss_fn
        
    def extract_data_from_batch(self, batch: dict, train: bool) -> tuple[torch.Tensor]:
        
        images_lst = []
        for image in self.trainer.datamodule.images:
            images_lst.append(batch[image]["data"])
        ANAT = batch["ANAT"]["data"]
        for l in torch.unique(ANAT)[1:]:
            images_lst.append(torch.where(ANAT==l, 1, 0))
        images = torch.cat(images_lst, dim=1)
        
        labels_lst = []
        for label in self.trainer.datamodule.labels:
            labels_lst.append(batch[label]["data"])
        labels = torch.cat(labels_lst, dim=1)

        return images, labels
    
    def on_test_start(self) -> None:
        raise NotImplementedError
    
            
            