import sys
import torch
import torch.nn as nn
from addict import Dict

# Ensure we can import from the PointTransformerV3 directory
# This assumes the docker container or env has it in PYTHONPATH
try:
    from model import PointTransformerV3
except ImportError:
    # Fallback for local testing if not in path
    # PointTransformerV3 is in /workspace/PointTransformerV3
    # We added /workspace to PYTHONPATH in docker-compose
    try:
        from PointTransformerV3.model import PointTransformerV3
    except ImportError as e:
        print(f"Warning: Could not import PointTransformerV3. Make sure PointTransformerV3 is in PYTHONPATH. Error: {e}")
        # Dummy class for syntax checking if import fails (don't rely on this at runtime)
        class PointTransformerV3(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()
            def forward(self, x): return x

class SharedPTv3Encoder(nn.Module):
    def __init__(self, input_channels=6, enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.backbone = PointTransformerV3(
            in_channels=input_channels,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            dec_depths=(2, 2, 2, 2), # Standard decoder config, though we might not use it all
            dec_channels=(64, 64, 128, 256),
            enable_flash=True, # Attempt to use flash attn
            cls_mode=False # We want dense features
        )
        
    def forward(self, data_dict):
        """
        Forward pass to get encoded features.
        Args:
            data_dict: Dict with 'coord', 'feat', 'grid_coord', 'offset'
        Returns:
            Point object with 'feat'
        """
        # The PTv3 forward method returns the decoder output by default if cls_mode=False
        # We need to consider if we want the ENCODER output or the DECODER output.
        # For segmentation, we usually want the decoder output (per-point features).
        # For MAE, we might intercept the encoder.
        
        # PTv3 forward: embedding -> enc -> dec
        output = self.backbone(data_dict)
        return output
