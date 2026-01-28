from cvdseg.lightning_modules.base import Base
from cvdseg.lightning_modules.ensemble import Ensemble
import torch
import torch.nn as nn

class ClassConditional(Base): 
    
    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        pred = self.network(x, i, **self.network_kwargs)
        spatial_dims = len(x.shape[2:])
        if len(pred.shape) == 2 + spatial_dims:
            return pred
        else:
            return torch.unbind(pred, dim=1)
    
    def training_step(self, batch: dict) -> torch.Tensor:
        images, labels = self.extract_data_from_batch(batch, train=True)
        losses = []
        for i in range(labels.shape[1]):
            if not torch.any(torch.isnan(labels[:,i])):
                pred = self(images, i)
                loss = self.criterion(pred, labels[:,i:i+1])
                losses.append(loss)
        loss = torch.mean(loss)
        self.log("loss", loss.item(), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: dict) -> None:
        images, labels = self.extract_data_from_batch(batch, train=False)
        preds_raw = []
        for i in range(labels.shape[1]):
            net = lambda x: self(x, i)
            preds_raw.append(self.test_inferer(images, net))
        preds_raw = torch.cat(preds_raw, dim=1)
        pred = self.test_post_processor(self, preds_raw)
        self.dice_metric(pred, labels)
        return pred

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
    
    
class ClassConditionalEnsemble(Ensemble):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensemble_preds = []
        for networks, temp_scaling_lst in zip(self.ensemble, self.temp_scaling):
            network_preds = []
            for i, (network, images, labels, inferer, temp) in enumerate(zip(networks, self.input_images_lst, self.output_labels_lst, self.inferers_lst, temp_scaling_lst)):
                inp = self.extract_required_input_channels(x, images)
                preds = []
                for class_i in range(len(labels)):
                    net = lambda x: network(x, class_i)
                    preds.append(inferer(inp, net, **self.network_kwargs[i]) / temp)
                pred = torch.cat(preds, dim=1)
                network_preds.append(pred)
            ensemble_preds.append(self.do_activation(torch.cat(network_preds, dim=1), labels))
        return torch.mean(torch.stack(ensemble_preds, dim=0), dim=0)