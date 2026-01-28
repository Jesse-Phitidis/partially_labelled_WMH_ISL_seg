from monai.networks.nets.dynunet import DynUNet, DynUNetSkipLayer
from monai.networks.blocks.dynunet_block import UnetBasicBlock, get_conv_layer, UnetOutBlock, UnetResBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from typing import List, Optional, Sequence, Union


class ClassConditionalUnetBasicBlock(UnetBasicBlock):
    def forward(self, x, i):
        return super().forward(x)
    
class ClassConditionalUnetResBlock(UnetResBlock):
    def forward(self, x, i):
        return super().forward(x)

class ClassConditionalUnetOutBlock(nn.Module):
    
    def __init__(self, spatial_dims, in_channels, out_channels, n_classes):
        super().__init__()
        self.emb = nn.ModuleList()
        for _ in range(n_classes):
            self.emb.append(UnetOutBlock(spatial_dims, in_channels, out_channels))
        
    def forward(self, x, i):
        return self.emb[i](x)
    

class ClassConditionalDynUNetSkipLayer(DynUNetSkipLayer):

    def forward(self, x, i):
        downout = self.downsample(x, i)
        nextout = self.next_layer(downout, i)
        upout = self.upsample(nextout, downout)
        if self.super_head is not None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout, i)
        return upout
    

DynUNet.__init__.__globals__["DynUNetSkipLayer"] = ClassConditionalDynUNetSkipLayer
DynUNet.__init__.__globals__["UnetBasicBlock"] = ClassConditionalUnetBasicBlock
DynUNet.__init__.__globals__["UnetResBlock"] = ClassConditionalUnetResBlock

class ClassConditionalDynUNet(DynUNet):
    
    '''
    forward takes the input tensor and an index. Each index uses a different 1x1x1 conv head for the output.
    '''
    
    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.recursive_replace(self, self.emb_replace_fn)
        
    def forward(self, x, i):
        out = self.skip_layers(x, i)
        out = self.output_block(out, i)
        if self.training and self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(interpolate(feature_map, out.shape[2:]))
            return torch.stack(out_all, dim=1)
        return out
    
    def recursive_replace(self, current_module, replace_fn: callable, top_module=None, **replace_fn_kwargs):
    
        if top_module is None:
            top_module = current_module

        for child_name, child_module in current_module.named_children():
            new_child = replace_fn(top_module, child_module, **replace_fn_kwargs)
            
            if new_child is not None:
                setattr(current_module, child_name, new_child)
            
            self.recursive_replace(child_module, replace_fn, top_module, **replace_fn_kwargs)
            
    def emb_replace_fn(self, top_module, child_module):
        if isinstance(child_module, UnetOutBlock):
            
            w_device, w_dtype = child_module.conv.conv.weight.device, child_module.conv.conv.weight.dtype
            b_device, b_dtype = child_module.conv.conv.bias.device, child_module.conv.conv.bias.dtype
            
            new_module = ClassConditionalUnetOutBlock(top_module.spatial_dims, child_module.conv.conv.in_channels, top_module.out_channels, self.n_classes)
            
            for i in range(self.n_classes):
                new_weight = nn.Parameter(torch.empty(new_module.emb[i].conv.conv.weight.shape, device=w_device, dtype=w_dtype))
                new_bias = nn.Parameter(torch.empty(new_module.emb[i].conv.conv.bias.shape, device=b_device, dtype=b_dtype))
                
                # This is how pytorch initializes the weights and biases of a Conv3d module
                nn.init.kaiming_normal_(new_weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(new_bias, -bound, bound)
                    
                setattr(new_module.emb[i].conv.conv, 'weight', new_weight)
                setattr(new_module.emb[i].conv.conv, 'bias', new_bias)
            
            return new_module
        else:
            return None