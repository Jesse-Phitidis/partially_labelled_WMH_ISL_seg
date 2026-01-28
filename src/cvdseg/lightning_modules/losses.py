import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss, DeepSupervisionLoss
from monai.utils import DiceCEReduction, look_up_option, pytorch_after
from typing import Sequence, Callable
import copy
import warnings


class BrierLoss(nn.Module):
    def __init__(self, topk=False):
        super().__init__()
        self.topk = (topk / 100) if topk is not False else False
    def forward(self, pred, target):
        mse = nn.functional.mse_loss(pred, target, reduction="none")
        if self.topk is False:
            return torch.mean(mse)
        mse = torch.flatten(mse, start_dim=1)
        k = round(self.topk * mse.shape[1])
        topk = torch.topk(mse, k=k, dim=1)
        return torch.mean(topk.values)


class MyBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    '''
    This class provides one improvement:
    
    1. The input weights do not need to be tensors already.
    '''
    def __init__(self, pos_weight: Sequence | None = None, *args, **kwargs):
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        super().__init__(pos_weight=pos_weight, *args, **kwargs)


class MyDiceCELoss(DiceCELoss):
    
    '''
    This class provides three improvements:
    
    1. The input weights do not need to be tensors already.
    2. The option to normalise given weights in a sensible way.
    3. The option to force BCE to be used for multilabel settings. 
    '''
    
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        multilabel: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        dice_weight: Sequence | None = None,
        ce_weight: Sequence | None = None,
        sensible_weight_norm: bool = False,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` and ``weight`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``. only used by the `DiceLoss`, not for the `CrossEntropyLoss` and `BCEWithLogitsLoss`.
            multilabel: use nn.BCEWithLogitsLoss for each channel to accomodate a multilabel setting.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            dice_weight: the weights for monai.losses.DiceCELoss. If include_background=False then it should not include
                a value for the first channel.
            ce_weight: the weight param for torch.nn.CrossEntropyLoss or the pos_weight param for torch.nn.BCEWithLogitsLoss 
                if the prediction is single channel or if multilabel=True.
            sensible_weight_norm: normalise given weights so they sum to the number of classes (after possibly removing 
                background for DiceLoss).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.multilabel = multilabel
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.sensible_weight_norm = sensible_weight_norm

        if dice_weight is not None:
            dice_weight = torch.tensor(dice_weight)
        if ce_weight is not None:
            ce_weight = torch.tensor(ce_weight)
        if dice_weight is not None and sensible_weight_norm:
            assert len(dice_weight) != 1, "Cannot use sensible_weight_norm for Dice on a single channel"
            dice_weight = (dice_weight / dice_weight.sum()) * len(dice_weight)
        if ce_weight is not None and sensible_weight_norm:
            assert len(ce_weight) != 1, "Cannot use sensible weight norm with BCE since background per channel is always given weight of 1"
            ce_weight = (ce_weight / ce_weight.sum()) * len(ce_weight)
        else:
            dice_weight = dice_weight
            ce_weight = ce_weight

        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=dice_weight,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.binary_cross_entropy = nn.BCEWithLogitsLoss(pos_weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        if input.shape[1] == 1 or self.multilabel:
            ce_loss = self.bce(input, target)
        else:
            ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
    
    
class ClassAdaptiveMyDiceCELoss(MyDiceCELoss):

    '''
    Calculates loss only for available labels
    '''
    
    def __init__(self, multilabel=False, *args, **kwargs):
        if multilabel:
            print("multilabel must be False - setting to False")
        super().__init__(multilabel=False, *args, **kwargs)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if input.shape[1] < 3:
            raise ValueError(
                f"PartiallyLabelledDiceCELoss requires at least 3 output channels (bg and two or more fg)"
                )
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
            
        # Get original weights or set to correct length of ones if these were none
        n_channels = input.shape[1]
        if self.dice.weight is None:
            dice_weight = torch.ones(n_channels if self.dice.include_background else n_channels - 1)
        else:
            dice_weight = copy.deepcopy(self.dice.weight)
        if self.cross_entropy.weight is None:
            ce_weight = torch.ones(n_channels)
        else:
            ce_weight = copy.deepcopy(self.cross_entropy.weight)
        
        # Loop over batch since different nan channels could be present
        total_loss = torch.tensor(0.0, device=input.device)
        n_batches = target.shape[0]
        for b in range(n_batches):
            b_input = input[b:b+1]
            b_target = target[b:b+1]
            # Set to zero for channels where target is nan and also for background if nan channels are present
            nan_channels = torch.unique(torch.where(torch.isnan(b_target))[1])
            for c in nan_channels:
                if (not self.dice.include_background) and (c==0):
                    pass
                else:
                    dice_weight[c - (0 if self.dice.include_background else 1)] = 0.0
                ce_weight[c] = 0.0
                b_target[:,c] = 0.0 # If we leave it as nans the weight of 0 makes no difference since 0 * torch.nan = torch.nan
            
            # Define losses in local scope
            dice = copy.deepcopy(self.dice)
            dice.weight = dice_weight
            ce = copy.deepcopy(self.cross_entropy)
            ce.weight = ce_weight.to(b_input.device)

            # Compute loss
            dice_loss = dice(b_input, b_target)
            ce_loss = self.ce(b_input, b_target, ce)
            total_loss += 1/n_batches * (self.lambda_dice * dice_loss + self.lambda_ce * ce_loss)

        return total_loss
    
    def ce(self, input: torch.Tensor, target: torch.Tensor, fn: nn.Module) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return fn(input, target)  # type: ignore[no-any-return]


class MarginalMyDiceCELoss(MyDiceCELoss):

    '''
    Merges all missing labels into a single background label (and sums the post softmax channel predictions).
    It is necessary to apply softmax and then log separately here, so we will see if this causes any numerical issues.
    '''
    
    def __init__(
        self, 
        sigmoid=False,
        softmax=False, 
        other_act=None,
        multilabel=False, 
        average_weights=False,
        *args, 
        **kwargs):
        if sigmoid:
            print("sigmoid must be False - setting to False")
        if softmax:
            print("softmax must be False - setting to False")
        if other_act is not None:
            print("other_act must be None - setting to None")
        if multilabel:
            print("multilabel must be False - setting to False")
        super().__init__(
            sigmoid=False,
            softmax=False,
            other_act=None,
            multilabel=False, 
            *args, 
            **kwargs)
        if average_weights:
            self.weight_merge_fn = torch.mean
        else:
            self.weight_merge_fn = torch.sum
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if input.shape[1] < 3:
            raise ValueError(
                f"PartiallyLabelledDiceCELoss requires at least 3 output channels (bg and two or more fg)"
                )
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
            
        # Get original weights or set to correct length of ones if these were none
        n_channels = input.shape[1]
        if self.dice.weight is None:
            dice_weight = torch.ones(n_channels if self.dice.include_background else n_channels - 1)
        else:
            dice_weight = copy.deepcopy(self.dice.weight)
        if self.cross_entropy.weight is None:
            ce_weight = torch.ones(n_channels)
        else:
            ce_weight = copy.deepcopy(self.cross_entropy.weight)
        
        # Loop over batch since different nan channels could be present
        total_loss = torch.tensor(0.0, device=input.device)
        n_batches = target.shape[0]
        for b in range(n_batches):
            
            b_dice_weight = copy.deepcopy(dice_weight)
            b_ce_weight = copy.deepcopy(ce_weight)
            
            b_input = input[b:b+1]
            b_target = target[b:b+1]
            
            nan_channels = torch.unique(torch.where(torch.isnan(b_target))[1]).tolist()
            
            if len(nan_channels) != 0:
                
                # We must include bg so set weight to mean channel weight e.g. [1,1] --> [1,1,1]
                if not self.dice.include_background: 
                    b_dice_weight = torch.cat([torch.mean(b_dice_weight, dim=0, keepdim=True), b_dice_weight], dim=0)
                    if self.sensible_weight_norm:
                        b_dice_weight = (b_dice_weight / b_dice_weight.sum()) * len(b_dice_weight)
                
                not_nan_channels = [i for i in range(b_target.shape[1]) if i not in nan_channels]
                
                # Merge targets as the complement of available (mutually exclusive) labels
                merged_bg_target = 1 - torch.max(b_target[:, not_nan_channels], dim=1, keepdim=True).values
                b_target = torch.cat([merged_bg_target, b_target[:, not_nan_channels]], dim=1)
                
                # Merge inputs. Sum AFTER softmax. Original paper github seems not to do this, but I think its wrong.
                b_input = torch.softmax(b_input, dim=1)
                merged_bg_input = torch.sum(b_input[:, nan_channels], dim=1, keepdim=True)
                b_input = torch.cat([merged_bg_input, b_input[:, not_nan_channels]], dim=1)
                
                # Merge weights as mean or sum (makes more sense)
                merged_bg_dice_weight = self.weight_merge_fn(b_dice_weight[nan_channels], dim=0, keepdim=True)
                b_dice_weight = torch.cat([merged_bg_dice_weight, b_dice_weight[not_nan_channels]])
                merged_bg_ce_weight = self.weight_merge_fn(b_ce_weight[nan_channels], dim=0, keepdim=True)
                b_ce_weight = torch.cat([merged_bg_ce_weight, b_ce_weight[not_nan_channels]])
                
            else:
                b_input = torch.softmax(b_input, dim=1)
            
            # Define losses in local scope
            dice = copy.deepcopy(self.dice)
            dice.weight = b_dice_weight
            if not self.dice.include_background and len(nan_channels) != 0:
                dice.include_background = True
            reduction = self.cross_entropy.reduction
            ce = torch.nn.NLLLoss(weight=b_ce_weight.to(b_input.device), reduction=reduction)

            # Compute loss
            dice_loss = dice(b_input, b_target)
            ce_loss = self.ce(torch.log(b_input + 1e-8), torch.argmax(b_target, dim=1, keepdim=True), ce)
            total_loss += 1/n_batches * (self.lambda_dice * dice_loss + self.lambda_ce * ce_loss)

        return total_loss
    
    def ce(self, input: torch.Tensor, target: torch.Tensor, fn: nn.Module) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return fn(input, target)  # type: ignore[no-any-return]


MyDeepSupervisionLoss_DefaultConfig = {
    "weight_mode": "exp",
    "norm": True
}

class MyDeepSupervisionLoss(DeepSupervisionLoss):
    
    '''
    Add an argument to normalise the weights to sum to one like for nnU-Net
    '''
    
    def __init__(self, norm: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        
    def get_weights(self, *args, **kwargs) -> list[float]:
        weights = super().get_weights(*args, **kwargs)
        if not self.norm:
            return weights
        return (np.array(weights) / np.sum(weights)).tolist()