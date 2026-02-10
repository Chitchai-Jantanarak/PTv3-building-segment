# src/models/mae/masking.py

import torch
from torch import Tensor


class BlockMasking:
    def __init__(
        self,
        ratio: float = 0.75,
        block_size: int = 64,
    ):
        self.ratio = ratio
        self.block_size = block_size

    def __call__(
        self,
        coord: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = coord.device
        unique_batches = torch.unique(batch)

        visible_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=device)
        masked_mask = torch.zeros(coord.shape[0], dtype=torch.bool, device=device)

        for b in unique_batches:
            batch_mask = batch == b
            batch_coord = coord[batch_mask]

            block_ids = self._assign_blocks(batch_coord)
            unique_blocks = torch.unique(block_ids)
            n_blocks = unique_blocks.shape[0]

            n_masked = int(n_blocks * self.ratio)
            perm = torch.randperm(n_blocks, device=device)
            masked_blocks = unique_blocks[perm[:n_masked]]
            visible_blocks = unique_blocks[perm[n_masked:]]

            batch_indices = torch.where(batch_mask)[0]

            for blk in masked_blocks:
                blk_mask = block_ids == blk
                masked_mask[batch_indices[blk_mask]] = True

            for blk in visible_blocks:
                blk_mask = block_ids == blk
                visible_mask[batch_indices[blk_mask]] = True

        visible_indices = torch.where(visible_mask)[0]
        masked_indices = torch.where(masked_mask)[0]

        return visible_indices, masked_indices, visible_mask

    def _assign_blocks(self, coord: Tensor) -> Tensor:
        block_coord = torch.floor(coord / self.block_size).long()

        mins = block_coord.min(dim=0)[0]
        block_coord = block_coord - mins

        dims = block_coord.max(dim=0)[0] + 1
        block_ids = (
            block_coord[:, 0] * dims[1] * dims[2]
            + block_coord[:, 1] * dims[2]
            + block_coord[:, 2]
        )

        return block_ids


def random_masking(
    n_points: int,
    ratio: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    n_masked = int(n_points * ratio)
    perm = torch.randperm(n_points, device=device)
    masked_indices = perm[:n_masked]
    visible_indices = perm[n_masked:]
    return visible_indices, masked_indices
