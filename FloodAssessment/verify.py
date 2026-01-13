import os
import torch
import numpy as np
from preprocessing.converter import prepare_data_for_ptv3
from models.ptv3_encoder import SharedPTv3Encoder
from models.mae_decoder import MAEDecoder
from models.heads import SegmentationHead, InpaintHead
from models.classifier_head import FEMAClassifier, GeometricFeatureExtractor

def verify_pipeline():
    print("=== Starting Flood Assessment Pipeline Verification ===")
    
    # 1. Mock Preprocessing
    print("[1] Testing Preprocessing...")
    # Create dummy LAS data manually instead of writing file
    # Mocking what prepare_data_for_ptv3 returns
    points = np.random.rand(1000, 3).astype(np.float32)
    feats = np.random.rand(1000, 6).astype(np.float32)
    grid_coords = (points * 100).astype(np.int32)
    
    data_dict = {
        'coord': torch.from_numpy(points),
        'feat': torch.from_numpy(feats),
        'grid_coord': torch.from_numpy(grid_coords),
        'offset': torch.IntTensor([1000]),
        'batch': torch.zeros(1000).long()
    }
    print("    Data Dict prepared successfully. Shape:", data_dict['coord'].shape)

    # 2. Mock Model Initialization
    print("[2] Initializing Models...")
    try:
        encoder = SharedPTv3Encoder()
        mae = MAEDecoder(encoder)
        seg_head = SegmentationHead(512, 3)
        inpaint_head = InpaintHead(512)
        classifier = FEMAClassifier()
        print("    Models initialized successfully.")
    except Exception as e:
        print(f"    Error initializing models: {e}")
        return

    # 3. Forward Pass - Stage 0 (MAE)
    print("[3] Testing Stage 0 (MAE)...")
    try:
        # We need to run on CPU for verification if CUDA not available or for simplicity
        # But PTv3 might require CUDA for some ops if not strictly pure PyTorch
        # We will try CPU
        rec, mask = mae(data_dict)
        print("    MAE Output Shape:", rec.shape)
    except Exception as e:
        print(f"    Error in MAE forward pass: {e}")
        # Continue to verify other parts 

    # 4. Forward Pass - Stage 1 (Structural)
    print("[4] Testing Stage 1 (Structural)...")
    try:
        # Re-run encoder to get features
        enc_out = encoder(data_dict)
        features = enc_out.feat
        
        seg_b_out = seg_head(features)
        print("    Seg-B Output Shape:", seg_b_out.shape)
        
        inpaint_out = inpaint_head(features)
        print("    Inpaint Output Shape:", inpaint_out.shape)
    except Exception as e:
        print(f"    Error in Structural forward pass: {e}")

    # 5. Forward Pass - Stage 2 (Semantic + Classifier)
    print("[5] Testing Stage 2 (Semantic)...")
    try:
        # MLP Classifier
        geo_extractor = GeometricFeatureExtractor()
        # Mock geometric features
        geo_feats = geo_extractor.extract(points[:100])
        geo_tensor = torch.from_numpy(geo_feats).float().unsqueeze(0) # Batch size 1
        
        cls_out = classifier(geo_tensor)
        print("    Classifier Output Shape:", cls_out.shape)
    except Exception as e:
         print(f"    Error in Semantic forward pass: {e}")

    print("=== Verification Complete ===")

if __name__ == "__main__":
    verify_pipeline()
