import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .train_mae import PointDataset, collate_fn
from models.ptv3_encoder import SharedPTv3Encoder
from models.heads import SegmentationHead, InpaintHead

def train_structural_pipeline():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)['structural']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Stage 1: Structural Training on {device}")
    
    # Components
    encoder = SharedPTv3Encoder()
    # Load Pretrained MAE
    try:
        encoder.load_state_dict(torch.load(cfg['model']['pretrained_encoder'], map_location=device))
        print("Loaded MAE Pretrained Encoder")
    except FileNotFoundError:
        print("Warning: Pretrained encoder not found, starting from scratch")
        
    if cfg['model']['freeze_encoder']:
        for param in encoder.parameters():
            param.requires_grad = False
            
    # Heads
    # Seg-B: Terrain(0), Building(1), Ignore(2)
    seg_b = SegmentationHead(in_channels=512, num_classes=3).to(device)
    inpaint = InpaintHead(in_channels=512).to(device)
    
    optimizer = optim.AdamW([
        {'params': seg_b.parameters()},
        {'params': inpaint.parameters()}
    ], lr=cfg['training']['lr'])
    
    criterion_seg = nn.CrossEntropyLoss(ignore_index=2)
    criterion_regr = nn.MSELoss()
    
    # Dataloader
    dataset = PointDataset(cfg['data']['root'], cfg['data']['voxel_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], collate_fn=collate_fn, shuffle=True)
    
    # Loop
    encoder.to(device)
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        for batch in dataloader:
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
