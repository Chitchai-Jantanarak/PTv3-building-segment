import torch
from torch import Tensor


def block_local_knn(
    query_coord: Tensor,
    ref_coord: Tensor,
    ref_features: Tensor,
    query_batch: Tensor | None = None,
    ref_batch: Tensor | None = None,
    k: int = 8,
    block_size: float = 0.05,
    chunk_size: int = 2048,
) -> Tensor:
    N = query_coord.shape[0]
    D = ref_features.shape[1]
    device = query_coord.device

    output = torch.zeros(N, D, device=device, dtype=ref_features.dtype)

    has_batch = query_batch is not None and ref_batch is not None

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        query_chunk = query_coord[start:end]
        C = query_chunk.shape[0]

        if not has_batch:
            diff = (query_chunk.unsqueeze(1) - ref_coord.unsqueeze(0)).abs()
            in_box = diff.max(dim=-1).values < block_size

            for i in range(C):
                local_mask = in_box[i]
                n_local = local_mask.sum().item()

                if n_local == 0:
                    dist_all = ((query_chunk[i] - ref_coord) ** 2).sum(dim=-1)
                    nearest = dist_all.argmin()
                    output[start + i] = ref_features[nearest]
                    continue

                local_features = ref_features[local_mask]
                local_coords = ref_coord[local_mask]

                k_actual = min(k, n_local)
                dists = ((query_chunk[i] - local_coords) ** 2).sum(dim=-1)
                topk_dists, topk_idx = dists.topk(k_actual, largest=False)

                weights = 1.0 / (topk_dists + 1e-6)
                weights = weights / weights.sum()

                output[start + i] = (
                    local_features[topk_idx] * weights.unsqueeze(-1)
                ).sum(dim=0)
        else:
            query_batch_chunk = query_batch[start:end]

            for b in range(int(ref_batch.max().item()) + 1):
                ref_mask_b = ref_batch == b
                if not ref_mask_b.any():
                    continue

                ref_coord_b = ref_coord[ref_mask_b]
                ref_feat_b = ref_features[ref_mask_b]

                for i in range(C):
                    if query_batch_chunk[i].item() != b:
                        continue

                    diff = (query_chunk[i] - ref_coord_b).abs()
                    in_box = diff.max(dim=-1).values < block_size

                    if not in_box.any():
                        dist_all = ((query_chunk[i] - ref_coord_b) ** 2).sum(dim=-1)
                        nearest = dist_all.argmin()
                        output[start + i] = ref_feat_b[nearest]
                        continue

                    local_features = ref_feat_b[in_box]
                    local_coords = ref_coord_b[in_box]

                    k_actual = min(k, local_features.shape[0])
                    dists = ((query_chunk[i] - local_coords) ** 2).sum(dim=-1)
                    topk_dists, topk_idx = dists.topk(k_actual, largest=False)

                    weights = 1.0 / (topk_dists + 1e-6)
                    weights = weights / weights.sum()

                    output[start + i] = (
                        local_features[topk_idx] * weights.unsqueeze(-1)
                    ).sum(dim=0)

    return output
