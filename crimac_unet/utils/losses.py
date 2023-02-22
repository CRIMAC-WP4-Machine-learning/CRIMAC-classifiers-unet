""""
Copyright 2021 the Norwegian Computing Center

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _WeightedLoss
from pytorch_toolbelt.losses import DiceLoss as DiceLoss_toolbelt


class FocalLoss(nn.modules.loss._WeightedLoss):
    '''
    The multi-class focal loss comes from:
    https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    '''

    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
'''
The following implementation of the multi-class dice loss has been modified from:
https://github.com/Guocode/DiceLoss.Pytorch/blob/master/loss.py 
'''

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(predict.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target)
                if self.weight is not None:
                    assert self.weight.shape[0] == predict.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class CombinedCEFocalLoss(nn.Module):
    '''
    A combination of focal loss and weighted cross entropy loss for volumetric outputs and ground truths

    The following implementation of the combined loss has been modified from:
    https://discuss.pytorch.org/t/extending-multi-class-2d-dice-loss-to-3d/59666/2
    '''
    def __init__(self, weight, gamma, t1=0.8, t2=1.2):
        super(CombinedCEFocalLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)
        self.focal_loss = FocalLoss(weight=weight, gamma=gamma)
        self.t1 = t1
        self.t2 = t2

    def forward(self, output, ground_truth):
        """
        Forward pass

        :param output: torch.tensor (NxCxDxHxW) Network output (logits) not normalized.
        :param ground_truth: torch.tensor (NxDxHxW)
        :return: scalar
       """
        y_1 = self.cross_entropy_loss.forward(output, ground_truth)
        y_2 = self.focal_loss.forward(output, ground_truth)

        return self.t1*y_1 + self.t2*y_2


class DiceCELoss(_WeightedLoss):
    def __init__(self, weight, lambda_ce=1, lambda_dice=1, ignore_index=-100):
        """
        This loss computes the weighted sum of _weighted_ softdiceloss and _weighted_ CE loss.
        The function fulfills the following requirements:
            - multiclass dice loss
            - weighting of classes
            - ignore_index = -100

        I re-implemented this since
        - nnUnet DC_and_CE_loss does not support weighting
        - Monai DiceCELoss does not support ignore_index

        Args:
            weights: class weights for ce and softdiceloss separately
            lambda_ce (int, optional): weight for the CE loss term
            lambda_dice (int, optional): weight for the DICE loss term
            ignore_index (int, optional): _description_. Defaults to -100.
        """        
        
        # I used DiceLoss from pytorch_toolbelt since segmentation_models_pytorch DiceLoss
        # has weird dependencies (installs pytorch 1.7.1, https://github.com/qubvel/segmentation_models.pytorch/issues/703)
        super(DiceCELoss, self).__init__(weight=weight)
        
        # scale weights to 0, 1
        self.weight /= self.weight.sum()

        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.ce = nn.CrossEntropyLoss(weight=self.weight, reduction='mean') 
        # note: for CrossEntropyLoss, passing `weight`` when `reduction = "mean"` scales weights to [0, 1]
        
        
        self.dice = DiceLoss_toolbelt(mode='multiclass', ignore_index=ignore_index)
        self.dice.aggregate_loss = lambda x: (x * self.weight).mean() # reduction = 'mean' (weighted)
        
        
    def forward(self, y_pred, y_true):
        loss_dice = self.dice(y_pred, y_true)
        loss_ce = self.ce(y_pred, y_true) 
        return self.lambda_ce * loss_ce + self.lambda_dice * loss_dice


