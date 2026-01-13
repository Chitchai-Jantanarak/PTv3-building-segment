import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

class GeometricFeatureExtractor:
    """
    Extracts geometric features from a point cloud cluster.
    Input: Points (N, 3) of a single building instance.
    Output: Feature vector (32,)
    """
    def extract(self, points, mae_error=0.0):
        if len(points) < 4:
            return np.zeros(32)
            
        points = np.asarray(points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # [0] Height
        z_min, z_max = np.min(z), np.max(z)
        height = z_max - z_min
        
        # [1] Footprint Area (Approximation via Convex Hull or BBox)
        # Using simple AABB for now for speed
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        dims = bbox_max - bbox_min
        footprint_area = dims[0] * dims[1]
        
        # [2] Volume
        volume = footprint_area * height
        
        # [3-6] BBox
        bbox_x, bbox_y, bbox_z = dims
        
        # [29] Global Curvature (PCA)
        try:
            pca = PCA(n_components=3)
            pca.fit(points)
            eigenvalues = pca.explained_variance_
            # Curvature ~ min_eigenvalue / sum(eigenvalues)
            curvature = eigenvalues[2] / (np.sum(eigenvalues) + 1e-6)
        except:
             curvature = 0
             
        # ... Implement other calculations as needed ...
        
        features = np.zeros(32)
        features[0] = height
        features[1] = footprint_area
        features[2] = volume
        features[3] = bbox_x
        features[4] = bbox_y
        features[5] = bbox_z
        features[29] = curvature
        features[30] = mae_error
        
        return features

class FEMAClassifier(nn.Module):
    def __init__(self, input_dim=32, num_classes=28): # 28 FEMA codes
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)
