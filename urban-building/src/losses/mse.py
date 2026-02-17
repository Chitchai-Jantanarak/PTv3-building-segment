# src/losses/mse.py
import torch
import torch.nn as nn
from torch import Tensor


def point_mse_loss(
    pred: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    diff = pred - target
    mse = torch.sum(diff**2, dim=-1)

    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    return mse


def masked_mse_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    diff = pred - target
    mse = torch.mean(diff**2, dim=-1)

    masked_mse = mse * mask.float()

    if reduction == "mean":
        n_masked = mask.sum()
        if n_masked > 0:
            return masked_mse.sum() / n_masked
        return torch.tensor(0.0, device=pred.device)
    elif reduction == "sum":
        return masked_mse.sum()
    return masked_mse


class PointMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return point_mse_loss(pred, target, self.reduction)


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return masked_mse_loss(pred, target, mask, self.reduction)
