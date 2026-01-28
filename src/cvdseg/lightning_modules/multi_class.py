from cvdseg.lightning_modules.base import Base 
import torch
import torch.nn as nn
from typing import Callable

class Multiclass(Base): 

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
        
        if train:
            fg = torch.max(torch.cat(labels_lst, dim=1), dim=1, keepdim=True).values
            bg = 1 - fg
            labels_lst.insert(0, bg)
            
        labels = torch.cat(labels_lst, dim=1)

        return images, labels
    
    def on_test_start(self) -> None:
        raise NotImplementedError