import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, features):
        return self.head(features)

class InpaintHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Takes features of building points
        # Outputs offset/displacement or new points
        # Simplified: Regress to "corrected" Z or Normal
        self.head = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output: Adjusted XYZ or Offset
        )
        
    def forward(self, features):
        return self.head(features)
