import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import time
import torch
from typing import Any
from cvdseg.lightning_modules import constants as C
import wandb
import numpy as np
import random
from monai.metrics import DiceMetric

class TimeIteration(Callback):

    """Average time to process a batch"""

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.train_dataloader_len = len(trainer.train_dataloader)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch_time = time.perf_counter() - pl_module.epoch_start_time
        mean_iter_time = epoch_time / pl_module.train_dataloader_len
        pl_module.log("mean_iter_time", mean_iter_time, on_step=False, on_epoch=True)
        
        
class LogImagesCallback(Callback):
    
    def __init__(
        self, 
        pred_images: list[str],
        pred_labels: list[list[str]],
        batch_images: list[str],
        batch_labels: list[list[str]],
        pred: bool = True,
        pred_freq: int = 5,
        pred_frac: float = 1.0,
        pred_shuffle: bool = False, 
        pred_slc: str = "mid", 
        batch: bool = True,
        batch_freq: str | int = "once", 
        batch_frac: float = 0.25,
        batch_slc: str = "mid", 
        ) -> None:
        
        '''
        Args:
            pred_images: List of sequences to plot, e.g. ["T1w", "FLAIR", "T2w"]
            pred_labels: List of lists with overlays to show for each sequences, e.g. [["WMH", "ISLtissue"], ["WMH", "ISLtissue"], ["PVS"]]
            batch_images: Same as for pred_images
            batch_labels: Same as for pred_images except only the ground truth is shown, not the prediction
            pred: Whether to plot tables for the predictions
            pred_freq: How many validation steps before logging predictions
            pred_frac: What fraction of the validation data to show
            pred_shuffle: Whether to shuffle the data. If True, different validation samples may be shown each time
            pred_slc: Either "mid" to plot the middle slice or name of a label to plot the slice with most of this label
            batch: Whether to plot tables for the batch. If pred and batch are both false this callback does nothing
            batch_freq: How many training epochs per plotting of batch. If set to "once" will  only plot for epoch 1
            batch_frac: Fraction of training set to plot
            batch_slc: Same as for pred_slc
            
        '''
        
        super().__init__()
        
        assert pred_slc in ([l for sl in pred_labels for l in sl] + ["mid"])
        assert batch_slc in ([l for sl in batch_labels for l in sl] + ["mid"])
        
        self.pred_images = pred_images
        self.pred_labels = pred_labels
        self.batch_images = batch_images
        self.batch_labels = batch_labels
        self.pred = pred
        self.pred_freq = pred_freq
        self.pred_frac = pred_frac
        self.pred_shuffle = pred_shuffle
        self.pred_slc = pred_slc
        self.batch = batch
        if batch_freq == "once":
            self.batch_once = True
            self.batch_freq = 1
        else:
            self.batch_once = False
            self.batch_freq = batch_freq
        self.batch_frac = batch_frac
        self.batch_slc = batch_slc
        
        self.batch_data = []
        self.pred_data = []
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: torch.Tensor | dict[str, Any], batch: Any, batch_idx: int) -> None:
        self.process_batch(
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            is_batch=True
        )
        
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: torch.Tensor | dict[str, Any] | None, batch: Any, batch_idx: int) -> None:
        self.process_batch(
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            is_batch=False
        )
            
    def process_batch(self, trainer: pl.Trainer, outputs: torch.Tensor | dict[str, Any], batch: Any, is_batch: bool) -> None:
        if is_batch:
            mod_term = self.batch_freq
            setting = self.batch
            images = self.batch_images
            labels = self.batch_labels
            data = self.batch_data
        else:
            mod_term = self.pred_freq * trainer.check_val_every_n_epoch
            setting = self.pred
            images = self.pred_images
            labels = self.pred_labels
            data = self.pred_data
            
        if ((trainer.current_epoch + 1) % mod_term != 0) or (not setting) or (is_batch and self.batch_once and trainer.current_epoch + 1 > 1):
            return None
        
        slc_lst = self.get_slc(batch, is_batch=is_batch)
        stem_lst = batch[images[0]]["stem"]
        
        for i_batch in range(trainer.datamodule.batch_size):
            slc = slc_lst[i_batch].item()
            ID = stem_lst[i_batch].split("_sub")[0]
            row_data = [ID]
            scores_lst = []
            
            for image, label in zip(images, labels):
                index_to_class = {k: v for k, v in C.index_to_class.items() if v in label}
                img, lab, pred, scores = self.process_for_overlay(
                    trainer=trainer, 
                    batch=batch, 
                    image=image,
                    label=label, 
                    pred=outputs,
                    i_batch=i_batch,
                    slc=slc,
                    is_batch=is_batch
                )
                masks = {}
                if not is_batch:
                    masks["prediction"] = {"mask_data": pred, "class_labels": index_to_class}
                masks["ground truth"] = {"mask_data": lab, "class_labels": index_to_class}
                wandb_image = wandb.Image(img, masks=masks)
                row_data.append(wandb_image)
                scores_lst.append(scores)
            row_data[0] = row_data[0] + self.format_scores(scores_lst)
            data.append(row_data)
            
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if ((trainer.current_epoch + 1) % self.batch_freq != 0) or (not self.batch) or (self.batch_once and trainer.current_epoch + 1 > 1):
            return None            
        last_index = int(self.batch_frac * len(self.batch_data)) - 1
        data = self.batch_data[:last_index] # Would be faster to just stop at the right point but can't for val preds if shuffle on so keep consistent/general
        pl_module.logger.log_table(key=f"batches/{trainer.current_epoch}", columns=["ID", *self.batch_images], data=data)
        self.batch_data = []
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if ((trainer.current_epoch + 1) % (self.pred_freq * trainer.check_val_every_n_epoch) != 0) or (not self.pred):
            return None
        if self.pred_shuffle:
            random.shuffle(self.pred_data)        
        last_index = int(self.pred_frac * len(self.pred_data)) - 1
        data = self.pred_data[:last_index]
        pl_module.logger.log_table(key=f"preds/{trainer.current_epoch}", columns=["ID", *self.pred_images], data=data)
        self.pred_data = []
            
    def get_slc(self, batch: Any, is_batch: bool) -> int:
        method = self.batch_slc if is_batch else self.pred_slc
        
        shape = batch[self.batch_images[0]]["data"].shape
        mid_slc_int = shape[-1] // 2 # For is_batch=True this is half patch size but may be different for val
        mid_slc = torch.tensor(mid_slc_int).repeat(shape[0]).reshape(shape[0], 1)
        if method == "mid":
            return mid_slc
        
        label = batch[method]["data"]
        assert label.shape[1] == 1, "LogImagesCallback requires single channel per batch"
        reduced_xy = torch.sum(label, dim=(2,3))
        maxi, argmaxi = torch.max(reduced_xy, dim=-1)
        for i in range(shape[0]):
            if maxi[i] == 0:
                argmaxi[i] = mid_slc_int
        return argmaxi
    
    def process_for_overlay(self, trainer: pl.Trainer, batch: Any, image: str, label: str, pred: Any, i_batch: int, slc: int, is_batch: bool) -> tuple:
        
        img_out = batch[image]["data"][i_batch, 0, :, :, slc].cpu().numpy()
        lab_out = np.zeros_like(img_out)
        
        label_masks_for_score = []
        
        for lab in label:
            mask = batch[lab]["data"][i_batch, 0, :, :, slc].cpu().numpy().astype(bool)
            if np.sum(mask) > 0:
                assert np.max(lab_out[mask]) == 0, "Check that only mutually exclusive label are being overlayed in LogImagesCallback"
            label_masks_for_score.append(mask[None, None, ...])
            lab_out[mask] = C.class_to_index[lab]
            
        if is_batch:
            return img_out, lab_out, None, None
        
        pred_out = np.zeros_like(img_out)
        
        pred_masks_for_score = []
        
        for channel_out, lab in zip(torch.unbind(pred, dim=1), trainer.datamodule.labels):
            if lab not in label:
                continue
            mask = channel_out[i_batch, :, :, slc].cpu().numpy().astype(bool)
            if np.sum(mask) > 0:
                assert np.max(pred_out[mask]) == 0, "Check that only mutually exclusive preds are being overlayed in LogImagesCallback"
            pred_masks_for_score.append(mask[None, None, ...])
            pred_out[mask] = C.class_to_index[lab]
            
        metric = DiceMetric(include_background=True, ignore_empty=False)
        scores_dict = {}
        for gt, pred, lab in zip(label_masks_for_score, pred_masks_for_score, label):
            score = metric(torch.tensor(pred), torch.tensor(gt))
            scores_dict[lab] = score.item()
            
        return img_out, lab_out, pred_out, scores_dict
    
    def format_scores(self, scores: list) -> str:
        if scores[0] is None:
            return ""
        
        new_dict = {}
        for d in scores:
            for lab, score in d.items():
                if lab not in new_dict:
                    new_dict[lab] = score
        
        string = ""
        for lab, score in new_dict.items():
            if np.isnan(score):
                num = "NaN"
            else:
                num = str(int(np.round(100*score, 0))) + "%"
            string += f"\n{lab}: {num}"
        return string
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def process_for_overlay(self, trainer: pl.Trainer, batch: Any, image: str, label: str, pred: Any, i_batch: int, slc: int, is_batch: bool) -> tuple:
        
    #     img_out = batch[image]["data"][i_batch, 0, :, :, slc].cpu().numpy()
    #     lab_out = np.zeros_like(img_out)
        
    #     for lab in label:
    #         mask = batch[lab]["data"][i_batch, 0, :, :, slc].cpu().numpy().astype(bool)
    #         if np.sum(mask) > 0:
    #             assert np.max(lab_out[mask]) == 0, "Check that only mutually exclusive label are being overlayed in LogImagesCallback"
    #         lab_out[mask] = C.class_to_index[lab]
            
    #     if is_batch:
    #         return img_out, lab_out, None
        
    #     pred_out = np.zeros_like(img_out)
        
    #     for channel_out, lab in zip(torch.unbind(pred, dim=1), trainer.datamodule.labels):
    #         if lab not in label:
    #             continue
    #         mask = channel_out[i_batch, :, :, slc].cpu().numpy().astype(bool)
    #         if np.sum(mask) > 0:
    #             assert np.max(pred_out[mask]) == 0, "Check that only mutually exclusive preds are being overlayed in LogImagesCallback"
    #         pred_out[mask] = C.class_to_index[lab]
            
    #     return img_out, lab_out, pred_out
        
        