import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import sys
import multiprocessing

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import correct function (reusing logic is good)
from preprocessing.converter import prepare_data_for_ptv3

def process_single_file(args):
    """
    Wrapper for multiprocessing.
    """
    fpath, save_path, voxel_size = args
    if os.path.exists(save_path):
        return # Skip if done
    
    try:
        # PTV3 converter handles H5 now via updated io_utils
        data_dict, _ = prepare_data_for_ptv3(fpath, voxel_size=voxel_size)
        
        # Save as standard torch dict
        torch.save(data_dict, save_path)
        return True
    except Exception as e:
        print(f"Failed to process {os.path.basename(fpath)}: {e}")
        return False

def preprocess_whu_dataset(src_root, dst_root, voxel_size=0.04, num_workers=4):
    """
    Recursively finds H5 files in src_root (checking for ALS, MLS, MLS-W structures)
    and processes them into dst_root.
    """
    os.makedirs(dst_root, exist_ok=True)
    
    # 1. Find all H5 files recursively
    # WHU structure: ALS/*.h5, MLS/*/*.h5 etc.
    print(f"Scanning {src_root} for .h5 files...")
    all_files = glob.glob(os.path.join(src_root, "**", "*.h5"), recursive=True)
    
    if not all_files:
        print(f"No .h5 files found in {src_root}. Check your path.")
        return

    print(f"Found {len(all_files)} files.")
    print(f"Processing to {dst_root} with voxel_size={voxel_size}...")
    
    tasks = []
    for fpath in all_files:
        # preserve relative structure if desired? 
        # or flat? The user requested simple offline processing. 
        # Flat is easiest for DataLoader usually, unless names collide.
        
        fname = os.path.basename(fpath)
        save_name = os.path.splitext(fname)[0] + ".pth"
        
        # Check subfolder logic if names collide
        # For now assume unique names or flattening
        save_path = os.path.join(dst_root, save_name)
        tasks.append((fpath, save_path, voxel_size))
        
    # Parallelize
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks)))
    else:
        for t in tqdm(tasks):
            process_single_file(t)

if __name__ == "__main__":
    # User requested placeholders they can change
    # Default assumption: mounted at /workspace/data or similar
    
    # Placeholder paths - User should edit these
    TRAIN_SRC = "data/raw/whu" 
    TRAIN_DST = "data/processed/whu"
    
    # Optional: Separate Test/Val if needed
    # TEST_SRC = "data/raw/whu_test"
    # TEST_DST = "data/processed/whu_test"
    
    VOXEL_SIZE = 0.04
    WORKERS = 4
    
    print("Starting WHU-Urban3D Preprocessing...")
    print(f"Source: {TRAIN_SRC}")
    print(f"Dest:   {TRAIN_DST}")
    
    preprocess_whu_dataset(TRAIN_SRC, TRAIN_DST, voxel_size=VOXEL_SIZE, num_workers=WORKERS)
    
    print("Done! Update config.yaml to point to 'data/processed/whu' and load .pth files.")
