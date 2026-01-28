import torch
import torch.nn as nn
from monai.networks import one_hot
from monai.utils import MetricReduction, look_up_option
from typing import Any
import importlib
from typing import Callable
from monai.networks.blocks.dynunet_block import UnetOutBlock
import math

def inverse_sigmoid(y: float) -> float:
    y = torch.as_tensor(y)
    return -torch.log((1 - y) / y)


def split_join(string, delim, slc):
    return (delim).join(string.split(delim)[slc])


def instantiate_object_from_config(config: dict, **kwargs) -> object:
        class_path_split = config["class_path"].split(".")
        module_name = (".").join(class_path_split[:-1])
        class_name = class_path_split[-1]
        class_ref = getattr(importlib.import_module(module_name), class_name)
        return class_ref(**config["init_args"], **kwargs)
    
    
def safe_mean(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    x[torch.isinf(x)] = float("nan")
    return torch.nanmean(x, *args, **kwargs)


# Added f[torch.isinf(f)] = torch.nan
def my_do_metric_reduction(
    f: torch.Tensor, reduction: MetricReduction | str = MetricReduction.MEAN
) -> tuple[torch.Tensor | Any, torch.Tensor]:
    """
    This function is to do the metric reduction for calculated `not-nan` metrics of each sample's each class.
    The function also returns `not_nans`, which counts the number of not nans for the metric.

    Args:
        f: a tensor that contains the calculated metric scores per batch and
            per class. The first two dims should be batch and class.
        reduction: define the mode to reduce metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``.
            if "none", return the input f tensor and not_nans.

    Raises:
        ValueError: When ``reduction`` is not one of
            ["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].
    """

    # some elements might be Nan (if ground truth y was missing (zeros))
    # we need to account for it
    nans = torch.isnan(f)
    infs = torch.isinf(f)
    not_nans = ~nans & ~infs

    t_zero = torch.zeros(1, device=f.device, dtype=torch.float)
    reduction = look_up_option(reduction, MetricReduction)
    if reduction == MetricReduction.NONE:
        return f, not_nans.float()

    f[nans] = 0
    if reduction == MetricReduction.MEAN:
        # 2 steps, first, mean by channel (accounting for nans), then by batch
        not_nans = not_nans.sum(dim=1).float()
        f = torch.where(not_nans > 0, f.sum(dim=1).float() / not_nans, t_zero)  # channel average

        not_nans = (not_nans > 0).sum(dim=0).float()
        f = torch.where(not_nans > 0, f.sum(dim=0).float() / not_nans, t_zero)  # batch average

    elif reduction == MetricReduction.SUM:
        not_nans = not_nans.sum(dim=[0, 1]).float()
        f = torch.sum(f, dim=[0, 1])  # sum over the batch and channel dims
    elif reduction == MetricReduction.MEAN_BATCH:
        not_nans = not_nans.sum(dim=0).float()
        f = torch.where(not_nans > 0, f.sum(dim=0).float() / not_nans, t_zero)  # batch average
    elif reduction == MetricReduction.SUM_BATCH:
        not_nans = not_nans.sum(dim=0).float()
        f = f.sum(dim=0).float()  # the batch sum
    elif reduction == MetricReduction.MEAN_CHANNEL:
        not_nans = not_nans.sum(dim=1).float()
        f = torch.where(not_nans > 0, f.sum(dim=1).float() / not_nans, t_zero)  # channel average
    elif reduction == MetricReduction.SUM_CHANNEL:
        not_nans = not_nans.sum(dim=1).float()
        f = f.sum(dim=1).float()  # the channel sum
    elif reduction != MetricReduction.NONE:
        raise ValueError(
            f"Unsupported reduction: {reduction}, available options are "
            '["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].'
        )
    return f, not_nans


def my_compute_channel(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """"""
        '''
        This makes it so that labels of nans are not considered in the dice calculation (by default, they would be considered the same as 0)
        '''
        y_o = torch.sum(y)
        if y_o > 0:
            return (2.0 * torch.sum(torch.masked_select(y, y_pred))) / (y_o + torch.sum(y_pred))
        if self.ignore_empty or torch.isnan(y_o):
            return torch.tensor(float("nan"), device=y_o.device)
        denorm = y_o + torch.sum(y_pred)
        if denorm <= 0:
            return torch.tensor(1.0, device=y_o.device)
        return torch.tensor(0.0, device=y_o.device)
    
    
def recursive_replace(current_module: nn.Module, replace_fn: Callable, top_module: None | nn.Module = None, **replace_fn_kwargs):
    
    if top_module is None:
        top_module = current_module

    for child_name, child_module in current_module.named_children():
        new_child = replace_fn(top_module, child_module, **replace_fn_kwargs)
        
        if new_child is not None:
            setattr(current_module, child_name, new_child)
        
        recursive_replace(child_module, replace_fn, top_module, **replace_fn_kwargs)
        
        
def out_channels_replace_fn(top_module: nn.Module, child_module: nn.Module, out_channels: int):
    
    if isinstance(child_module, UnetOutBlock):
        
        w_device, w_dtype = child_module.conv.conv.weight.device, child_module.conv.conv.weight.dtype
        b_device, b_dtype = child_module.conv.conv.bias.device, child_module.conv.conv.bias.dtype
        
        new_module = UnetOutBlock(top_module.spatial_dims, child_module.conv.conv.in_channels, out_channels, top_module.dropout)
        
        new_weight = nn.Parameter(torch.empty(new_module.conv.conv.weight.shape, device=w_device, dtype=w_dtype))
        new_bias = nn.Parameter(torch.empty(new_module.conv.conv.bias.shape, device=b_device, dtype=b_dtype))
        
        # This is how pytorch initializes the weights and biases of a Conv3d module
        nn.init.kaiming_normal_(new_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)
            
        setattr(new_module.conv.conv, 'weight', new_weight)
        setattr(new_module.conv.conv, 'bias', new_bias)
        
        return new_module
    
    else:
        return None