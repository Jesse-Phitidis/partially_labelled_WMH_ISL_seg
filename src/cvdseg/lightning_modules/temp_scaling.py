import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable


nll_criterion = nn.BCEWithLogitsLoss()

@torch.enable_grad() # need to set inference_mode=False in the trainer for this to work
def find_temp(network: nn.Module, loader: DataLoader, image_keys: list, label_keys: list, **network_kwargs) -> float:
    
    assert len(label_keys) == 1
    
    network = network.to(torch.device("cuda"))
        
    logits_lst = []
    gt_lst = []

    for batch in loader:
        
        images_lst = []
        for image in image_keys:
            images_lst.append(batch[image]["data"])
        images = torch.cat(images_lst, dim=1)
        
        labels_lst = []
        for label in label_keys:
            labels_lst.append(batch[label]["data"])
        labels = torch.cat(labels_lst, dim=1)
        
        images = images.to(torch.device("cuda"))
        labels = labels.to(torch.device("cuda"))
        
        with torch.no_grad():    
            logits_lst.append(network(images, **network_kwargs).detach().cpu())
        gt_lst.append(labels.cpu())
        
    logits = torch.cat(logits_lst, dim=0)
    gt = torch.cat(gt_lst, dim=0)
    
    ori_nll = nll_criterion(logits, gt)
    
    temp = nn.Parameter(torch.ones(1, device=torch.device("cpu")))
    optimiser = torch.optim.LBFGS([temp], lr=0.01, max_iter=10000)
    
    def eval():
        optimiser.zero_grad()
        loss = nll_criterion(logits / temp, gt)
        loss.backward()
        return loss
    
    optimiser.step(eval)
    
    new_nll = nll_criterion(logits / temp, gt)
    
    return temp.cpu().item(), ori_nll, new_nll
            
            
            
            
            
                
            