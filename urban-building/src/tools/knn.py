# src/tools/knn.py

import torch
from torch import Tensor


def block_local_knn(
    query_coord: Tensor,      # (N_q, 3) normalized [0,1]
    ref_coord: Tensor,        # (N_r, 3) normalized [0,1]
    ref_features: Tensor,     # (N_r, D)
    query_batch: Tensor | None = None,
    ref_batch: Tensor | None = None,
    k: int = 8,
    block_size: float = 0.05,
    chunk_size: int = 2048,  
) -> Tensor:
    N_q = query_coord.shape[0]
    D   = ref_features.shape[1]
    device = query_coord.device
    dtype  = ref_features.dtype

    if N_q == 0 or ref_coord.shape[0] == 0:
        return torch.zeros(N_q, D, device=device, dtype=dtype)

    has_batch = (
        query_batch is not None
        and ref_batch is not None
        and query_batch.numel() > 0
        and ref_batch.numel() > 0
    )

    if has_batch:
        return _batched_knn(
            query_coord, ref_coord, ref_features,
            query_batch, ref_batch,
            k, block_size, chunk_size,
        )
    else:
        return _single_knn(
            query_coord, ref_coord, ref_features,
            k, block_size, chunk_size,
        )


def _single_knn(
    query_coord: Tensor,   # (N_q, 3)
    ref_coord: Tensor,     # (N_r, 3)
    ref_features: Tensor,  # (N_r, D)
    k: int,
    block_size: float,
    chunk_size: int,
) -> Tensor:
    N_q, D = query_coord.shape[0], ref_features.shape[1]
    device, dtype = query_coord.device, ref_features.dtype
    output = torch.zeros(N_q, D, device=device, dtype=dtype)

    for start in range(0, N_q, chunk_size):
        end   = min(start + chunk_size, N_q)
        q     = query_coord[start:end]     # (C, 3)
        C     = q.shape[0]

        diff  = (q.unsqueeze(1) - ref_coord.unsqueeze(0)).abs()  # (C, N_r, 3)
        in_box = diff.amax(dim=-1) < block_size                   # (C, N_r) bool
        l2 = (q.unsqueeze(1) - ref_coord.unsqueeze(0)).pow(2).sum(-1)  # (C, N_r)

        l2_masked = l2.masked_fill(~in_box, float("inf"))              # (C, N_r)

        all_inf = (~in_box).all(dim=-1)                                 # (C,) bool
        if all_inf.any():
            l2_masked[all_inf] = l2[all_inf]   # use unmasked distances

        k_actual = min(k, ref_coord.shape[0])
        topk_dists, topk_idx = l2_masked.topk(k_actual, dim=1, largest=False)
                                                                        # (C, k)

        weights = 1.0 / (topk_dists + 1e-6)                            # (C, k)
        weights = weights / weights.sum(dim=1, keepdim=True)            # (C, k)
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)         # (C, k, D)
        neighbor_feats = ref_features.unsqueeze(0).expand(C, -1, -1)    # (C, N_r, D)
        neighbor_feats = neighbor_feats.gather(1, idx_expanded)          # (C, k, D)

        output[start:end] = (neighbor_feats * weights.unsqueeze(-1)).sum(dim=1)

    return output


def _batched_knn(
    query_coord: Tensor,
    ref_coord: Tensor,
    ref_features: Tensor,
    query_batch: Tensor,
    ref_batch: Tensor,
    k: int,
    block_size: float,
    chunk_size: int,
) -> Tensor:
    N_q = query_coord.shape[0]
    D   = ref_features.shape[1]
    device, dtype = query_coord.device, ref_features.dtype
    output = torch.zeros(N_q, D, device=device, dtype=dtype)

    batch_ids = query_batch.unique()

    for b in batch_ids:
        q_mask = query_batch == b
        r_mask = ref_batch   == b

        if not q_mask.any() or not r_mask.any():
            continue

        out_b = _single_knn(
            query_coord=query_coord[q_mask],
            ref_coord=ref_coord[r_mask],
            ref_features=ref_features[r_mask],
            k=k,
            block_size=block_size,
            chunk_size=chunk_size,
        )
        output[q_mask] = out_b

    return output