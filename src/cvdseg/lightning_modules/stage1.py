from cvdseg.lightning_modules.multi_class import Multiclass 
from cvdseg.lightning_modules import utils
import torch

class Stage1(Multiclass): 
    
    def extract_data_from_batch(self, batch: dict, train: bool) -> tuple[torch.Tensor]:
        
        images_lst = []
        for image in self.trainer.datamodule.images:
            images_lst.append(batch[image]["data"])
        images = torch.cat(images_lst, dim=1)
        
        labels_lst = []
        for label in self.trainer.datamodule.labels:
            labels_lst.append(batch[label]["data"])
        
        new_labels_lst = [] 
        labels = torch.cat(labels_lst, dim=1)
        labels[torch.isnan(labels)] = 0
        fg = torch.max(labels, dim=1, keepdim=True).values
        if train:
            bg = 1 - fg
            new_labels_lst.append(bg)
        new_labels_lst.append(fg)    
           
        labels = torch.cat(new_labels_lst, dim=1)

        return images, labels
    
    def on_validation_epoch_end(self) -> None:
        
            name = self.trainer.datamodule.labels[0]
            for label in self.trainer.datamodule.labels[1:]:
                name += ("_" + label)
            joint_labels = [name]
        
            dice = self.dice_metric.aggregate()
            self.dice_metric.reset()
            val_dict = {}
            for i, lab in enumerate(joint_labels):
                val_dict[f"dice/{lab}"] = dice[i].item()
            val_dict["dice/mean"] = utils.safe_mean(dice).item()

            self.log_dict(val_dict, on_step=False, on_epoch=True)