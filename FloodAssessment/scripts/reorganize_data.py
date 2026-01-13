import os
import zipfile
import shutil
import glob

def reorganize_data():
    base_dir = "e:/Lesson/2568-2/040613703_aidev/DoingSomeWeirdLikeWeirdo/FloodAssessment/data"
    raw_dir = os.path.join(base_dir, "raw")
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        print(f"Created {raw_dir}")
        
    # 1. Unzip all .zip files
    zip_files = glob.glob(os.path.join(base_dir, "*.zip"))
    print(f"Found {len(zip_files)} zip files.")
    
    for zip_path in zip_files:
        try:
            print(f"Unzipping {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is a bad zip file.")
        except Exception as e:
            print(f"Error unzipping {zip_path}: {e}")
            
    # 2. Move existing .ply files in base_dir to raw_dir
    ply_files = glob.glob(os.path.join(base_dir, "*.ply"))
    print(f"Found {len(ply_files)} PLY files in base directory.")
    
    for ply_path in ply_files:
        dest_path = os.path.join(raw_dir, os.path.basename(ply_path))
        if not os.path.exists(dest_path):
            print(f"Moving {os.path.basename(ply_path)} to raw/...")
            shutil.move(ply_path, dest_path)
        else:
            print(f"Skipping {os.path.basename(ply_path)}, already exists in raw/.")
            
    print("Reorganization complete.")
    
    # Check what we have in raw
    files_in_raw = os.listdir(raw_dir)
    print(f"Total files in {raw_dir}: {len(files_in_raw)}")
    if len(files_in_raw) > 0:
        print(f"Sample files: {files_in_raw[:5]}")

if __name__ == "__main__":
    reorganize_data()
