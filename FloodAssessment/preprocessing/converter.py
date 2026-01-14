import numpy as np
import torch
from .io_utils import read_point_cloud

def voxel_sample(coords, voxel_size):
    """
    Simple voxel sampling to reduce density and prepare grid coords.
    """
    grid_coords = np.round(coords / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
    return unique_indices, grid_coords

def prepare_data_for_ptv3(path, voxel_size=0.04, max_points=None):
    """
    Loads a LAS/PLY file and prepares the dictionary for PTv3 inference.
    
    Args:
        path (str): Path to PC file.
        voxel_size (float): Voxel size for grid sampling.
    
    Returns:
        dict: Data dictionary with 'coord', 'feat', 'grid_coord', 'offset'.
    """
    # 1. Load Data
    raw_data = read_point_cloud(path)
    points = raw_data['coord']
    
    # 2. Extract features (Color + Height + Intensity if available)
    # Default feature: RGB (normalized)
    feats = []
    if 'color' in raw_data:
        feats.append(raw_data['color'])
    else:
        # If no color, use constant ones or intensity
        feats.append(np.ones_like(points) * 0.5)
        
    if 'intensity' in raw_data:
        # Intensity usually needs normalization
        intensity = raw_data['intensity'].reshape(-1, 1).astype(np.float32)
        # Simple normalization assumes 16-bit or similar, better to standardize
        if intensity.max() > 255:
            intensity = intensity / 65535.0
        else:
            intensity = intensity / 255.0
        feats.append(intensity)
        
    # Append Height (Z) as explicit feature if requested (Relative height often useful)
    # feats.append(points[:, 2:3] - points[:, 2].min())
        
    features = np.hstack(feats).astype(np.float32)
    
    # 3. Voxelization / Grid Sampling
    # PTv3 usually operates on voxelized data to handle large scenes and define structure
    # However, for pure inference, we might want to keep all points or just voxelize for the model
    # The model expects 'grid_coord'
    
    # We shift coords to positive octant for simplicity in grid calculation usually
    coord_min = points.min(0)
    shifted_points = points - coord_min
    
    unique_idx, _ = voxel_sample(shifted_points, voxel_size)
    
    # Random Block Sampling (Best for MAE & Transfer Learning)
    # Preservation of local density is key for learning geometry.
    if max_points is not None and len(unique_idx) > max_points:
        # Get the voxelized points first to pick a center
        voxel_points = points[unique_idx]
        
        # Pick a random center point
        center_idx = np.random.randint(len(voxel_points))
        center_point = voxel_points[center_idx]
        
        # Block size: Let's assume ~50m block for large scenes or just crop distinct number of points via KNN/Radius?
        # Simpler: Just crop a box around center.
        block_size = 50.0 # meters
        
        # Define mask
        min_box = center_point - block_size / 2
        max_box = center_point + block_size / 2
        
        # Apply crop to the *sub-sampled* points (since we already voxelized)
        # points[unique_idx] are the representative points
        sub_p = points[unique_idx]
        
        mask = np.all((sub_p >= min_box) & (sub_p <= max_box), axis=1)
        crop_idx = unique_idx[mask]
        
        # If crop is still too big, random sample from it
        if len(crop_idx) > max_points:
             choice = np.random.choice(len(crop_idx), max_points, replace=False)
             crop_idx = crop_idx[choice]
        # If crop is too small (e.g. edge), we might want to pick another or just take what we have.
        # Fallback: if we have too few points (< 10% of max), just revert to random sampling to ensure stability
        elif len(crop_idx) < (max_points // 10):
             choice = np.random.choice(len(unique_idx), max_points, replace=False)
             crop_idx = unique_idx[choice]
             
        unique_idx = crop_idx
    
    sub_points = points[unique_idx]
    sub_feats = features[unique_idx]
    sub_grid_coords = np.round((sub_points - coord_min) / voxel_size).astype(np.int32)
    
    # 4. Create Batch/Offset
    # For a single sample, offset is just [num_points]
    offset = torch.IntTensor([len(sub_points)])
    
    # 5. Convert to Tensor
    data_dict = {
        'coord': torch.from_numpy(sub_points).float(),
        'feat': torch.from_numpy(sub_feats).float(),
        'grid_coord': torch.from_numpy(sub_grid_coords).int(),
        'offset': offset,
        'batch': torch.zeros(len(sub_points)).long() # Batch index 0
    }
    
    if 'raw_las' in raw_data:
        return data_dict, raw_data['raw_las'].header
    elif 'raw_ply' in raw_data:
        return data_dict, raw_data['raw_ply']
    
    return data_dict, None
