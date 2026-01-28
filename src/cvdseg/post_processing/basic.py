import torch
from cvdseg.lightning_modules import constants as C
import pytorch_lightning as pl
from monai.networks import one_hot
from cvdseg.lightning_modules import utils

class ThresholdedPostProcessor:
    
    def __init__(self, threshold: float, already_activated: bool = False) -> None:
        self.threshold = threshold
        self.sigmoid = (lambda x: x) if already_activated else torch.sigmoid
        
    def __call__(self, pl_module: pl.LightningModule, pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class MultilabelPostProcessor(ThresholdedPostProcessor):
    
    def __call__(self, pl_module: pl.LightningModule, pred: torch.Tensor) -> torch.Tensor:
            pred = self.sigmoid(pred)
            mutually_exclusive_channels = []
            independent_channels = []
            for i, label in enumerate(pl_module.trainer.datamodule.labels):
                if label in C.mutually_exclusive_labels:
                    mutually_exclusive_channels.append((pred[:, i:i+1, ...],i))
                else:
                    independent_channels.append((pred[:, i:i+1, ...],i))
                
            output = torch.zeros_like(pred)

            if len(mutually_exclusive_channels) > 0:
                mutually_exclusive_channels_output = self.compute_mutually_exclusive_preds(
                    torch.cat([t[0] for t in mutually_exclusive_channels], dim=1),
                    T=self.threshold
                )

            if len(independent_channels) > 0:
                independent_channels_output = torch.where(torch.cat([t[0] for t in independent_channels], dim=1) > self.threshold, 1, 0)

            for i, tup in enumerate(mutually_exclusive_channels):
                output[:, tup[1], ...] = mutually_exclusive_channels_output[:, i, ...]

            for i, tup in enumerate(independent_channels):
                output[:, tup[1], ...] = independent_channels_output[:, i, ...]

            return output
        
    def compute_mutually_exclusive_preds(self, pred: torch.Tensor, T:float) -> torch.Tensor:
        output = torch.zeros_like(pred)
        maxi, argmaxi = torch.max(pred, dim=1, keepdim=True)
        ones_mask = one_hot(argmaxi, num_classes=pred.shape[1], dim=1).to(bool)
        zero_mask = torch.where(maxi < T, 1, 0).expand_as(pred).to(bool)
        output[ones_mask & ~zero_mask] = 1
        return output
    

class MulticlassPostProcessor:
    
    def __call__(self, pl_module: pl.LightningModule, pred: torch.Tensor) -> torch.Tensor:
        return one_hot(torch.argmax(pred, dim=1, keepdim=True), num_classes=pred.shape[1], dim=1)[:, 1:, ...]
        

class MulticlassMultilabelPostProcessor(ThresholdedPostProcessor):
    
    def __call__(self, pl_module: pl.LightningModule, pred: torch.Tensor) -> torch.Tensor:
    
        mutually_exclusive_channels = []
        independent_channels = []
        mutually_exclusive_channels.append((pred[:, 0:1, ...],0))
        for i, label in enumerate(pl_module.trainer.datamodule.labels):
            i += 1
            if label in C.mutually_exclusive_labels:
                mutually_exclusive_channels.append((pred[:, i:i+1, ...],i))
            else:
                independent_channels.append((pred[:, i:i+1, ...],i))
            
        output = torch.zeros_like(pred)

        mutually_exclusive_channels_output = self.compute_mutually_exclusive_preds(
            torch.cat([t[0] for t in mutually_exclusive_channels], dim=1)
        )

        if len(independent_channels) != 0:
            independent_channels_output = torch.where(torch.cat([self.sigmoid(t[0]) for t in independent_channels], dim=1) > self.threshold, 1, 0)

        for i, tup in enumerate(mutually_exclusive_channels):
            output[:, tup[1], ...] = mutually_exclusive_channels_output[:, i, ...]

        for i, tup in enumerate(independent_channels):
            output[:, tup[1], ...] = independent_channels_output[:, i, ...]

        return output[:, 1:, ...]
    
    def compute_mutually_exclusive_preds(self, pred: torch.Tensor) -> torch.Tensor:
        return one_hot(torch.argmax(pred, dim=1, keepdim=True), num_classes=pred.shape[1], dim=1)