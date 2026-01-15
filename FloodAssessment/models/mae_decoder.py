import torch
import torch.nn as nn
from .ptv3_encoder import SharedPTv3Encoder

class MAEDecoder(nn.Module):
    def __init__(self, encoder, mask_ratio=0.6, enc_channels=64, target_channels=6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # Decoder for reconstruction (Lightweight)
        # Head 1: Geometry (XYZ) - Reconstruct coordinates or offsets
        self.geo_head = nn.Sequential(
            nn.Linear(enc_channels, 256),
            nn.GELU(),
            nn.Linear(256, 3) # XYZ
        )
        
        # Head 2: Color (RGB) - Reconstruct color values
        self.color_head = nn.Sequential(
            nn.Linear(enc_channels, 256),
            nn.GELU(),
            nn.Linear(256, 3) # RGB (normalized 0-1)
        )

    def mask_points(self, data_dict):
        """
        Randomly mask points/patches.
        """
        feat = data_dict['feat']
        batch_size = feat.shape[0] # Total points
        
        # Simple random mask (Simulating damage)
        # In a real "Inpainting" training loop, the data loader might provide 
        # the mask based on geometric cropping (removing walls).
        # Here we do random masking for pretraining compatibility.
        mask = torch.rand(batch_size, device=feat.device) < self.mask_ratio
        
        data_dict['mask'] = mask
        # We zero-out masked features (Token approach)
        data_dict['feat'][mask] = 0 
        
        return data_dict

    def forward(self, data_dict):
        # 1. Masking (Only during training if valid)
        if self.training:
            data_dict = self.mask_points(data_dict)
            
        # 2. Encode
        output = self.encoder(data_dict)
        features = output.feat
        
        # 3. Decode
        # Reconstruct Geometry (XYZ) and Color (RGB)
        pred_geo = self.geo_head(features)
        pred_color = self.color_head(features)
        
        return pred_geo, pred_color, data_dict.get('mask', None)
