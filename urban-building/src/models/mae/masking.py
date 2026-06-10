# src/models/mae/masking.py

import torch
from torch import Tensor


class BlockMasking:
    def __init__(
        self,
        ratio: float = 0.75,
        block_size: int = 64,
        min_visible: int = 64,
    ):
        self.ratio = ratio
        self.block_size = block_size
        self.min_visible = min_visible

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
            batch_indices = torch.where(batch_mask)[0]
            n_sample = batch_indices.shape[0]

            block_ids = self._assign_blocks(batch_coord)
            unique_blocks = torch.unique(block_ids)
            n_blocks = unique_blocks.shape[0]

            min_vis_blocks = max(1, n_blocks - int(n_blocks * self.ratio))
            if self.min_visible > 0 and n_sample > 0:
                pts_per_block = max(1, n_sample // n_blocks)
                needed_blocks = max(min_vis_blocks, -(-self.min_visible // pts_per_block))
                n_keep = min(n_blocks, needed_blocks)
            else:
                n_keep = min_vis_blocks
            n_masked = n_blocks - n_keep    

            perm = torch.randperm(n_blocks, device=device)
            masked_blocks = unique_blocks[perm[:n_masked]]
            visible_blocks = unique_blocks[perm[n_masked:]]

            masked_block_mask = torch.isin(block_ids, masked_blocks)
            visible_block_mask = torch.isin(block_ids, visible_blocks)

            masked_mask[batch_indices[masked_block_mask]] = True
            visible_mask[batch_indices[visible_block_mask]] = True

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


class RandomMasking:
    def __init__(
        self,
        ratio: float = 0.75,
        min_visible: int = 64,
    ):
        self.ratio = ratio
        self.min_visible = min_visible

    def __call__(
        self,
        coord: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = coord.device
        n = coord.shape[0]
        visible_mask = torch.zeros(n, dtype=torch.bool, device=device)

        for b in torch.unique(batch):
            batch_indices = torch.where(batch == b)[0]
            n_sample = batch_indices.shape[0]
            n_keep = max(
                min(self.min_visible, n_sample),
                n_sample - int(n_sample * self.ratio),
            )
            perm = torch.randperm(n_sample, device=device)
            visible_mask[batch_indices[perm[:n_keep]]] = True

        visible_indices = torch.where(visible_mask)[0]
        masked_indices = torch.where(~visible_mask)[0]

        return visible_indices, masked_indices, visible_mask


