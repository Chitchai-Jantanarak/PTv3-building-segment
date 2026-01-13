import os
import logging
import numpy as np
import laspy
import rasterio
from plyfile import PlyData

def read_ply(path, features=['red', 'green', 'blue']):
    """
    Reads a PLY file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    try:
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        x = np.asarray(vertex['x'])
        y = np.asarray(vertex['y'])
        z = np.asarray(vertex['z'])
        coords = np.vstack((x, y, z)).transpose()
        
        data = {'coord': coords, 'raw_ply': plydata}
        
        # Features
        # Check standard color names
        props = [p.name for p in vertex.properties]
        
        if 'red' in props and 'green' in props and 'blue' in props:
            r = np.asarray(vertex['red'])
            g = np.asarray(vertex['green'])
            b = np.asarray(vertex['blue'])
            
            # Normalize if uint8
            if r.max() > 1.0:
                r = r / 255.0
                g = g / 255.0
                b = b / 255.0
            
            data['color'] = np.vstack((r, g, b)).transpose()
            data['red'] = r
            data['green'] = g
            data['blue'] = b
            
        return data
    except Exception as e:
        logging.error(f"Error reading PLY file {path}: {e}")
        raise

def read_point_cloud(path):
    """
    Generic reader that dispatches based on extension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.las', '.laz']:
        return read_las(path)
    elif ext in ['.ply']:
        return read_ply(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def read_las(path, features=['red', 'green', 'blue', 'intensity']):
    """
    Reads a LAS/LAZ file and returns points and requested features.
    
    Args:
        path (str): Path to the LAS/LAZ file.
        features (list): List of feature names to extract (default: color + intensity).
        
    Returns:
        dict: Dictionary containing 'coord' (N, 3) and other requested features.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        las = laspy.read(path)
        
        # Coordinates are always required
        coords = np.vstack((las.x, las.y, las.z)).transpose()
        
        data = {'coord': coords, 'raw_las': las}
        
        # Extract features
        for feat in features:
            try:
                if feat in ['red', 'green', 'blue']:
                    # Normalize colors to [0, 1] if 16-bit
                    val = getattr(las, feat)
                    if np.max(val) > 255:
                        val = val / 65535.0
                    else:
                        val = val / 255.0
                    data[feat] = val
                else:
                    data[feat] = getattr(las, feat)
            except AttributeError:
                logging.warning(f"Feature '{feat}' not found in LAS file.")
        
        # Combine colors if all present
        if all(x in data for x in ['red', 'green', 'blue']):
            data['color'] = np.vstack((data['red'], data['green'], data['blue'])).transpose()
            
        return data
        
    except Exception as e:
        logging.error(f"Error reading LAS file {path}: {e}")
        raise

def write_las(path, coords, features=None, reference_las=None, scale=[0.01, 0.01, 0.01], offset=[0,0,0]):
    """
    Writes a LAS/LAZ file.
    
    Args:
        path (str): Output path.
        coords (np.ndarray): (N, 3) coordinates.
        features (dict): Dictionary of feature arrays (e.g. {'classification': ...}).
        reference_las (laspy.LasData): Optional reference to copy header info from.
        scale, offset: Header parameters if reference_las is not provided.
    """
    try:
        if reference_las:
            header = reference_las.header
            # Create new las based on reference version
            new_las = laspy.LasData(header)
        else:
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = scale
            header.offsets = offset
            new_las = laspy.LasData(header)

        new_las.x = coords[:, 0]
        new_las.y = coords[:, 1]
        new_las.z = coords[:, 2]
        
        if features:
            for params, values in features.items():
                # Handle special fields
                if params == 'color' and values.shape[1] == 3:
                     # Denormalize if they are 0-1 floats
                    if values.max() <= 1.0:
                        values = (values * 65535).astype(np.uint16)
                    new_las.red = values[:, 0]
                    new_las.green = values[:, 1]
                    new_las.blue = values[:, 2]
                elif hasattr(new_las, params):
                    setattr(new_las, params, values)
                else:
                    # Generic extra dimensions could be added here if needed
                    pass
                    
        new_las.write(path)
        logging.info(f"Saved LAS file to {path}")
        
    except Exception as e:
        logging.error(f"Error writing LAS file {path}: {e}")
        raise

def read_geotiff(path):
    """
    Reads a GeoTIFF file.
    
    Returns:
        tuple: (data, profile)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    try:
        with rasterio.open(path) as src:
            data = src.read()
            profile = src.profile
            return data, profile
    except Exception as e:
        logging.error(f"Error reading GeoTIFF {path}: {e}")
        raise
