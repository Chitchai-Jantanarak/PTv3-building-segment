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
    flat_pred = pred.view(-1, n_classes)
    flat_target = target.view(-1)

    # Compute pt from UNWEIGHTED cross-entropy so focal term is not distorted
    ce_unweighted = F.cross_entropy(
        flat_pred,
        flat_target,
        reduction="none",
        ignore_index=ignore_index,
    )

    pt = torch.exp(-ce_unweighted)
    focal_weight = (1 - pt) ** gamma

    # Apply class-weight alpha separately (per-sample, based on true class)
    if alpha is not None:
        # Clamp target to valid range for gathering (ignore_index -> 0, masked later)
        safe_target = flat_target.clamp(min=0, max=n_classes - 1)
        alpha_t = alpha.to(flat_pred.device).gather(0, safe_target)
        focal = alpha_t * focal_weight * ce_unweighted
    else:
        focal = focal_weight * ce_unweighted

    valid_mask = flat_target != ignore_index
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
