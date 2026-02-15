# src/losses/chamfer.py
import torch
import torch.nn as nn
from torch import Tensor


def _chunked_min_dist(source: Tensor, target: Tensor, chunk_size: int) -> Tensor:
    mins = []
    for i in range(0, source.shape[0], chunk_size):
        chunk = source[i : i + chunk_size]
        diff = chunk.unsqueeze(1) - target.unsqueeze(0)
        dist = (diff**2).sum(-1)
        mins.append(dist.min(dim=1)[0])
    return torch.cat(mins)


def chamfer_loss(
    pred: Tensor,
    target: Tensor,
    reduction: str = "mean",
    chunk_size: int = 2048,
) -> Tensor:
    min_pred_to_target = _chunked_min_dist(pred, target, chunk_size)
    min_target_to_pred = _chunked_min_dist(target, pred, chunk_size)

    if reduction == "mean":
        return min_pred_to_target.mean() + min_target_to_pred.mean()
    elif reduction == "sum":
        return min_pred_to_target.sum() + min_target_to_pred.sum()
    else:
        return min_pred_to_target, min_target_to_pred


class ChamferLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return chamfer_loss(pred, target, self.reduction)


def batch_chamfer_loss(
    pred: Tensor,
    target: Tensor,
    batch_pred: Tensor,
    batch_target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    unique_batches = torch.unique(batch_pred)
    losses = []

    for b in unique_batches:
        mask_pred = batch_pred == b
        mask_target = batch_target == b

        pred_b = pred[mask_pred]
        target_b = target[mask_target]

        if pred_b.shape[0] > 0 and target_b.shape[0] > 0:
            loss_b = chamfer_loss(pred_b, target_b, reduction="mean")
            losses.append(loss_b)

    if not losses:
        return torch.tensor(0.0, device=pred.device)

    if reduction == "mean":
        return torch.stack(losses).mean()
    elif reduction == "sum":
        return torch.stack(losses).sum()
    return torch.stack(losses)
