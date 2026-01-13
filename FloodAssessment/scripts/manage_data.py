import os
import zipfile
import math
import hashlib
import shutil
from pathlib import Path

# Configuration
CHUNK_SIZE_MB = 90
CHUNK_SIZE = CHUNK_SIZE_MB * 1024 * 1024
DATA_DIR = Path('data/raw/ply')
ZIP_DIR = Path('data/compressed')
TEMP_ZIP = Path('temp_full_data.zip')

def calculate_checksum(file_path):
    """Calculates SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def pack_data():
    """Compresses data directory into split zip files."""
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} does not exist.")
        return

    print(f"Creating compressed archive of {DATA_DIR}...")
    
    # 1. Create a single large zip file
    with zipfile.ZipFile(TEMP_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(DATA_DIR.parent)
                print(f"Adding {file_path} as {arcname}")
                zipf.write(file_path, arcname)

    # 2. Split the zip file
    if not ZIP_DIR.exists():
        ZIP_DIR.mkdir(parents=True)
    
    # Clean up old parts
    for f in ZIP_DIR.glob("data_part_*"):
        os.remove(f)

    file_size = TEMP_ZIP.stat().st_size
    num_chunks = math.ceil(file_size / CHUNK_SIZE)
    
    print(f"Splitting {file_size / (1024*1024):.2f} MB archive into {num_chunks} chunks of {CHUNK_SIZE_MB} MB...")

    with open(TEMP_ZIP, 'rb') as src:
        for i in range(num_chunks):
            chunk_name = ZIP_DIR / f"data_part_{i:03d}.bin"
            with open(chunk_name, 'wb') as dst:
                chunk_data = src.read(CHUNK_SIZE)
                dst.write(chunk_data)
            print(f"Created {chunk_name}")

    # 3. Clean up temp file
    os.remove(TEMP_ZIP)
    print("Packing complete. At 'data/compressed'.")

def unpack_data():
    """Reassembles and extracts the data."""
    if not ZIP_DIR.exists():
        print(f"Error: {ZIP_DIR} does not exist.")
        return

    # 1. Reassemble chunks
    parts = sorted(ZIP_DIR.glob("data_part_*.bin"))
    if not parts:
        print("No compressed data parts found.")
        return

    print(f"Found {len(parts)} chunks. Reassembling...")
    
    with open(TEMP_ZIP, 'wb') as dst:
        for part in parts:
            print(f"Reading {part}...")
            with open(part, 'rb') as src:
                shutil.copyfileobj(src, dst)

    # 2. Extract
    print(f"Extracting to {DATA_DIR.parent}...")
    try:
        with zipfile.ZipFile(TEMP_ZIP, 'r') as zipf:
            zipf.extractall(DATA_DIR.parent)
        print("Extraction successful.")
    except zipfile.BadZipFile:
        print("Error: Reassembled zip file is corrupted.")
    finally:
        if TEMP_ZIP.exists():
            os.remove(TEMP_ZIP)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python manage_data.py [pack|unpack]")
        return
    
    command = sys.argv[1]
    if command == 'pack':
        pack_data()
    elif command == 'unpack':
        unpack_data()
    else:
        print("Unknown command. Use 'pack' or 'unpack'.")

if __name__ == "__main__":
    main()
