import os
import logging
import subprocess

def convert_to_potree(las_path, output_dir, potree_converter_path="PotreeConverter"):
    """
    Converts a LAS file to Potree format using the PotreeConverter CLI.
    
    Args:
        las_path (str): Input LAS file.
        output_dir (str): Output directory for Potree octree.
        potree_converter_path (str): Path to the executable.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cmd = [
        potree_converter_path,
        las_path,
        "-o", output_dir,
        "--generate-page", "index"
    ]
    
    try:
        logging.info(f"Running PotreeConverter: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logging.info("Potree conversion successful.")
    except FileNotFoundError:
        logging.error("PotreeConverter executable not found. Please ensure it is in your PATH or provided.")
        logging.info("You can download it from: https://github.com/potree/PotreeConverter/releases")
    except subprocess.CalledProcessError as e:
        logging.error(f"PotreeConverter failed: {e}")
