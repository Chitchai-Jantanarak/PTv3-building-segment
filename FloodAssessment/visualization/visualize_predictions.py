import os
import argparse
import numpy as np
import laspy
import open3d as o3d
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_visual_las(output_path, coords, original_las=None, predictions=None, dummy=False):
    """
    Creates a LAS file with extra dimensions for visualization in CloudCompare.
    
    Args:
        output_path (str): Path to save the LAS file.
        coords (np.ndarray): (N, 3) coordinates.
        original_las (laspy.LasData): Original LAS data to copy header/colors from.
        predictions (dict): Dictionary of prediction arrays:
            - 'pred_class': (N,) integers (0=Terrain, 1=Building, 2=Ignore)
            - 'inpaint_delta': (N, 3) or (N, 1) displacement/value
        dummy (bool): If True, generates random data.
    """
    
    # 1. Setup Header
    if original_las:
        header = original_las.header
        new_las = laspy.LasData(header)
    else:
        # Create a default header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.01, 0.01, 0.01]
        header.offsets = [0, 0, 0]
        new_las = laspy.LasData(header)
        
    # 2. Assign Coordinates
    new_las.x = coords[:, 0]
    new_las.y = coords[:, 1]
    new_las.z = coords[:, 2]
    
    # 3. Copy Colors if available (and not overwritten by visualization)
    if original_las and hasattr(original_las, 'red'):
        new_las.red = original_las.red
        new_las.green = original_las.green
        new_las.blue = original_las.blue
    elif dummy:
        # Random colors
        new_las.red = np.random.randint(0, 65535, len(coords))
        new_las.green = np.random.randint(0, 65535, len(coords))
        new_las.blue = np.random.randint(0, 65535, len(coords))

    # 4. Add Extra Dimensions (Predictions)
    if predictions:
        # Define extra dimensions
        # CloudCompare handles 'extra_dims' automatically
        
        if 'pred_class' in predictions:
            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name="PredClass",
                type=np.uint8,
                description="Predicted Segmentation Class"
            ))
            new_las.PredClass = predictions['pred_class'].astype(np.uint8)
            
        if 'inpaint_delta' in predictions:
            # We usually visualize the Magnitude of the shift for a scalar field
            delta = predictions['inpaint_delta']
            if len(delta.shape) > 1 and delta.shape[1] == 3:
                magnitude = np.linalg.norm(delta, axis=1)
            else:
                magnitude = delta.flatten()
                
            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name="InpaintMag",
                type=np.float32,
                description="Inpainting Magnitude"
            ))
            new_las.InpaintMag = magnitude.astype(np.float32)

    new_las.write(output_path)
    logging.info(f"Saved visualization LAS to {output_path}")

def interactive_view(las_path):
    """
    Opens the LAS file in Open3D for quick review.
    """
    try:
        las = laspy.read(las_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color by Prediction if available, else RGB
        if hasattr(las, 'PredClass'):
            logging.info("Visualizing based on 'PredClass'...")
            # Map 0 (Terrain) -> Green, 1 (Building) -> Red, 2 -> Grey
            colors = np.zeros((len(points), 3))
            labels = las.PredClass
            
            colors[labels == 0] = [0.2, 0.8, 0.2] # Green
            colors[labels == 1] = [0.8, 0.2, 0.2] # Red
            colors[labels == 2] = [0.5, 0.5, 0.5] # Grey
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif hasattr(las, 'red'):
            logging.info("Visualizing based on RGB...")
            # Normalize to 0-1
            red = las.red / 65535.0
            green = las.green / 65535.0
            blue = las.blue / 65535.0
            rgb = np.vstack((red, green, blue)).transpose()
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            
        o3d.visualization.draw_geometries([pcd], window_name="Prediction Review")
        
    except Exception as e:
        logging.error(f"Failed to visualize with Open3D: {e}")
        logging.info("Note: Open3D might fail in headless environments (like inside Docker without X forwarding).")

def main():
    parser = argparse.ArgumentParser(description="Visualize Seg-B Predictions")
    parser.add_argument("--input", type=str, help="Path to input LAS file")
    parser.add_argument("--output", type=str, default="visualization_output.las", help="Path to output LAS file")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy data for testing")
    parser.add_argument("--view", action="store_true", help="Open Open3D viewer after generation")
    
    args = parser.parse_args()
    
    if args.dummy:
        logging.info("Generating dummy data...")
        num_points = 1000
        coords = np.random.rand(num_points, 3) * 100
        
        # Fake predictions
        # 0: Terrain, 1: Building
        pred_class = np.random.randint(0, 2, num_points)
        
        # Fake inpainting (only on buildings)
        inpaint_delta = np.zeros(num_points)
        inpaint_delta[pred_class == 1] = np.random.rand(np.sum(pred_class == 1)) * 5.0
        
        predictions = {
            'pred_class': pred_class,
            'inpaint_delta': inpaint_delta
        }
        
        create_visual_las(args.output, coords, predictions=predictions, dummy=True)
        
    else:
        if not args.input or not os.path.exists(args.input):
            logging.error("Input file required for non-dummy mode.")
            return

        # In a real workflow, you would load your model predictions here
        # For this standalone tool, we will just pass-through inputs or expect a 'prediction.npy' companion
        # For now, let's just warn
        logging.warning("Real inference loading not yet implemented. Use --dummy to test the pipeline logic.")
        # Placeholder: just copy input to output
        try:
            original = laspy.read(args.input)
            coords = np.vstack((original.x, original.y, original.z)).transpose()
            create_visual_las(args.output, coords, original_las=original)
        except Exception as e:
            logging.error(f"Error processing input: {e}")
            return

    if args.view:
        interactive_view(args.output)

if __name__ == "__main__":
    main()
