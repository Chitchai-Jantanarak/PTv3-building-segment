import sys
import os
import torch
import torch.nn as nn
from addict import Dict

# Dynamic path usage to find PointTransformerV3
ptv3_path = None
# Check common locations
possible_paths = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "PointTransformerV3"), # Inside project root
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "PointTransformerV3"), # Sibling to project
    "/workspace/PointTransformerV3",
    "/root/PointTransformerV3",
    "/root/model/PointTransformerV3",
    os.path.expanduser("~/PointTransformerV3")
]

for path in possible_paths:
    if os.path.exists(os.path.join(path, "model.py")):
        ptv3_path = path
        break

if ptv3_path:
    # Add the PARENT directory to sys.path so we can import 'PointTransformerV3' as a package
    # This is critical for relative imports inside the library to work
    parent_dir = os.path.dirname(ptv3_path)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    try:
        # Import as a package
        from PointTransformerV3.model import PointTransformerV3
    except ImportError as e:
        print(f"Error importing PointTransformerV3 from {ptv3_path}: {e}")
        sys.exit(1)
else:
    # Try standard import as last resort (if installed via pip)
    try:
        from PointTransformerV3.model import PointTransformerV3
    except ImportError:
        print("\n" + "="*60)
        print("CRITICAL ERROR: PointTransformerV3 library not found!")
        print("="*60)
        print(f"Checked the following locations:\n" + "\n".join(possible_paths))
        print("\nPlease clone the repository:")
        print("    git clone https://github.com/Pointcept/PointTransformerV3.git")
        print("="*60 + "\n")
        sys.exit(1)

class SharedPTv3Encoder(nn.Module):
    def __init__(self, input_channels=3, enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.backbone = PointTransformerV3(
            in_channels=input_channels,
            enc_depths=(2, 2, 2, 4, 2), # Reduced from (2, 2, 2, 6, 2)
            enc_channels=enc_channels,
            dec_depths=(1, 1, 1, 1), # Reduced from (2, 2, 2, 2)
            dec_channels=(64, 64, 128, 256),
            enable_flash=False, # Disable Flash Attn to prevent Core Dump on 5090/Nightly
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
