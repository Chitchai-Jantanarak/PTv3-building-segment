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
    
    # Optimization: Crop FIRST, then Voxelize.
    # Voxelizing 100M points takes forever. Cropping to a 50m block first is fast.
    
    # 3. Random Block Crop (Applied on Raw Points)
    if max_points is not None and len(points) > max_points:
        # Pick random center
        center_idx = np.random.randint(len(points))
        center_point = points[center_idx]
        
        block_size = 50.0 # meters
        
        min_box = center_point - block_size / 2
        max_box = center_point + block_size / 2
        
        # Fast boolean mask on raw numpy array
        mask = np.all((points >= min_box) & (points <= max_box), axis=1)
        
        # Safety: If crop is empty or too small, fallback to random sampling or just take original
        if np.sum(mask) < 1000:
             # Fallback: simple random indices
             choice_idx = np.random.choice(len(points), min(len(points), max_points * 5), replace=False)
             points = points[choice_idx]
             features = features[choice_idx]
        else:
             points = points[mask]
             features = features[mask]
             
    # 4. Voxelization / Grid Sampling (Now on smaller subset)
    coord_min = points.min(0)
    shifted_points = points - coord_min
    unique_idx, _ = voxel_sample(shifted_points, voxel_size)
    
    # 5. Final max_points check (if voxelization didn't reduce enough)
    if max_points is not None and len(unique_idx) > max_points:
        choice = np.random.choice(len(unique_idx), max_points, replace=False)
        unique_idx = unique_idx[choice]
        
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
