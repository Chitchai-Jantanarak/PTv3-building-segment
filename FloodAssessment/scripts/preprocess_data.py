import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.converter import prepare_data_for_ptv3

def preprocess_dataset(src_root, dst_root, voxel_size=0.04):
    """
    Reads all PLY/LAS files from src_root, processes them via prepare_data_for_ptv3,
    and saves the resulting dictionary as a .pth file in dst_root.
    """
    os.makedirs(dst_root, exist_ok=True)
    
    # gather files
    files = glob.glob(os.path.join(src_root, "*.ply")) + \
            glob.glob(os.path.join(src_root, "*.las")) + \
            glob.glob(os.path.join(src_root, "*.laz"))
            
    print(f"Found {len(files)} files in {src_root}")
    print(f"Processing to {dst_root} with voxel_size={voxel_size}...")
    
    for fpath in tqdm(files):
        fname = os.path.basename(fpath)
        save_name = os.path.splitext(fname)[0] + ".pth"
        save_path = os.path.join(dst_root, save_name)
        
        if os.path.exists(save_path):
            continue
            
        try:
            # We don't need the header for training data
            data_dict, _ = prepare_data_for_ptv3(fpath, voxel_size=voxel_size)
            torch.save(data_dict, save_path)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    # Config mirroring config.yaml
    TRAIN_SRC = "data/raw/ply/train"
    TRAIN_DST = "data/processed/train"
    
    # You might want to add validation/test paths here too if they exist
    # TEST_SRC = "data/raw/ply/test" 
    # TEST_DST = "data/processed/test"
    
    print("Starting Preprocessing...")
    preprocess_dataset(TRAIN_SRC, TRAIN_DST, voxel_size=0.04)
    # preprocess_dataset(TEST_SRC, TEST_DST, voxel_size=0.04)
    print("Done! Update config.yaml to point to 'data/processed/train' and load .pth files.")
