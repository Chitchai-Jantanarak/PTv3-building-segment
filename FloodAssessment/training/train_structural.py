import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.train_mae import PointDataset, collate_fn
from models.ptv3_encoder import SharedPTv3Encoder
from models.heads import SegmentationHead, InpaintHead

def train_structural_pipeline():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)['structural']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Stage 1: Structural Training on {device}")
    
    # Dataloader
    dataset = PointDataset(cfg['data']['root'], cfg['data']['voxel_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], collate_fn=collate_fn, shuffle=True)
    
    # Detect Input Dimension from first sample
    try:
        sample_file = dataset.files[0]
        sample_data = torch.load(sample_file)
        input_dim = sample_data['feat'].shape[1]
        print(f"Detected input feature dimension: {input_dim}")
    except Exception as e:
        print(f"Error detecting input dim, defaulting to 6: {e}")
        input_dim = 6

    # Components
    encoder = SharedPTv3Encoder(input_channels=input_dim)
    # Load Pretrained MAE
    pretrained_path = cfg['model']['pretrained_encoder']
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        # Adapt weights if channel mismatch exists (e.g. 3 vs 4)
        model_state = encoder.state_dict()
        
        # Check stem weight shape
        stem_key = "backbone.embedding.stem.conv.weight"
        if stem_key in checkpoint and stem_key in model_state:
            ckpt_w = checkpoint[stem_key]
            curr_w = model_state[stem_key]
            # Shape: [out, k, k, k, in] or [k,k,k,in,out]? 
            # Spconv SubMConv3d weight is usually [k, k, k, in, out] 
            # Error said: ckpt=[..., 3], curr=[..., 6]. Last dim mismatch.
            
            if ckpt_w.shape != curr_w.shape:
                print(f"Adapting checkpoint stem weights from {ckpt_w.shape} to {curr_w.shape}")
                # Assume last dim is input_channels
                c_in_ckpt = ckpt_w.shape[-1]
                c_in_curr = curr_w.shape[-1]
                
                new_w = curr_w.clone() # Keep random init for extra channels
                min_c = min(c_in_ckpt, c_in_curr)
                # Copy overlapping channels
                new_w[..., :min_c] = ckpt_w[..., :min_c]
                checkpoint[stem_key] = new_w
        
        encoder.load_state_dict(checkpoint, strict=False)
        print(f"Loaded MAE Pretrained Encoder from {pretrained_path}")
    except FileNotFoundError:
        print("Warning: Pretrained encoder not found, starting from scratch")
        
    if cfg['model']['freeze_encoder']:
        for param in encoder.parameters():
            param.requires_grad = False
            
    # Heads
    # Seg-B: Terrain(0), Building(1), Ignore(2)
    # Encoder output dim check: 64 (from dec_channels[0])
    seg_b = SegmentationHead(in_channels=64, num_classes=3).to(device)
    inpaint = InpaintHead(in_channels=64).to(device)
    
    optimizer = optim.AdamW([
        {'params': seg_b.parameters()},
        {'params': inpaint.parameters()}
    ], lr=cfg['training']['lr'])
    
    criterion_seg = nn.CrossEntropyLoss(ignore_index=2)
    criterion_regr = nn.MSELoss()
    
    # Loop
    encoder.to(device)
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        for batch in dataloader:
            if batch is None:
                continue
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            with torch.no_grad(): # Encoder frozen
                enc_out = encoder(batch)
                features = enc_out.feat
                
            # 1. Segmentation B
            seg_logits = seg_b(features)
            
            # Dummy labels for now as dataset might not have them
            # In real usage, batch['seg_label'] should exist
            if 'seg_label' in batch:
                target_seg = batch['seg_label']
            else:
                 # Dummy: random
                target_seg = torch.randint(0, 3, (features.shape[0],), device=device)
            
            loss_seg = criterion_seg(seg_logits, target_seg)
            
            # 2. Inpainting
            # Only on Building class (Index 1)
            # Use GT mask for training stability
            mask_building = target_seg == 1
            
            if mask_building.sum() > 0:
                inpaint_pred = inpaint(features[mask_building])
                # Target: e.g., offset data, or we assume input was sparse and we want to predict "missing" properties
                # For this dummy impl, we regress to 0 (stability) or actual geometry if available
                # Assuming 'target_geometry' exists
                target_geom = torch.zeros_like(inpaint_pred) 
                loss_inpaint = criterion_regr(inpaint_pred, target_geom)
            else:
                loss_inpaint = 0.0
                
            loss = loss_seg * cfg['training']['loss_weights']['seg_b'] + \
                   loss_inpaint * cfg['training']['loss_weights']['inpaint']
                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # Save
    torch.save({
        'encoder': encoder.state_dict(),
        'seg_b': seg_b.state_dict(),
        'inpaint': inpaint.state_dict()
    }, cfg['training']['save_path'])

if __name__ == "__main__":
    train_structural_pipeline()
