# src/losses/focal.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def focal_loss(
    pred: Tensor,
    target: Tensor,
    alpha: Optional[Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> Tensor:
    n_classes = pred.shape[-1]

    ce_loss = F.cross_entropy(
        pred.view(-1, n_classes),
        target.view(-1),
        weight=alpha,
        reduction="none",
        ignore_index=ignore_index,
    )

    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    focal = focal_weight * ce_loss

    valid_mask = target.view(-1) != ignore_index
    focal = focal[valid_mask]

    if reduction == "mean":
        return (
            focal.mean() if focal.numel() > 0 else torch.tensor(0.0, device=pred.device)
        )
    elif reduction == "sum":
        return focal.sum()
    return focal


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return focal_loss(
            pred,
            target,
            self.alpha,
            self.gamma,
            self.reduction,
            self.ignore_index,
        )


def dice_loss(
    pred: Tensor,
    target: Tensor,
    smooth: float = 1.0,
) -> Tensor:
    pred_soft = F.softmax(pred, dim=-1)
    n_classes = pred.shape[-1]

    target_one_hot = F.one_hot(target, n_classes).float()

    intersection = (pred_soft * target_one_hot).sum(dim=0)
    union = pred_soft.sum(dim=0) + target_one_hot.sum(dim=0)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()
