import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ptv3_encoder import SharedPTv3Encoder
from models.mae_decoder import MAEDecoder
from training.train_mae import PointDataset, collate_fn  # Reusing dataset utils

def train_inpaint_pipeline():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    # Basic config check (create dummy if missing for new script)
    if not os.path.exists(config_path):
        print("Config not found, using default params")
        cfg = {'inpaint': {'lr': 1e-4, 'epochs': 50, 'batch_size': 4, 'save_path': 'checkpoints/ptv3_inpaint.pth'},
               'data': {'root': 'data/processed'}}
    else:
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            cfg = full_cfg.get('inpaint', full_cfg.get('semantic')) # Fallback
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Stage 3: Inpainting (Seg-B) Training on {device}")
    
    # 1. Initialize Models
    encoder = SharedPTv3Encoder().to(device)
    # Seg-B uses MAE Decoder with Color Head
    decoder = MAEDecoder(encoder, mask_ratio=0.75, enc_channels=512).to(device)
    
    # Load Pretrained Backbone if available
    try:
        if os.path.exists("checkpoints/ptv3_mae.pth"):
            encoder.load_state_dict(torch.load("checkpoints/ptv3_mae.pth"), strict=False)
            print("Loaded pretrained backbone.")
    except Exception as e:
        print(f"Pretrained load warning: {e}")

    optimizer = optim.AdamW(decoder.parameters(), lr=float(cfg.get('lr', 1e-4)))
    
    # Losses
    # Chamfer Distance is ideal for geometry, but for "Token Masking" we use MSE on coordinates
    criterion_geo = nn.MSELoss() 
    criterion_color = nn.MSELoss()
    
    # Data
    dataset = PointDataset(cfg.get('data', {}).get('root', 'data'))
    dataloader = DataLoader(dataset, batch_size=int(cfg.get('batch_size', 4)), collate_fn=collate_fn)
    
    encoder.train()
    decoder.train()
    
    epochs = int(cfg.get('epochs', 10))
    
    for epoch in range(epochs):
        total_loss = 0
        total_geo = 0
        total_rgb = 0
        
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward Pass (Auto-Masking inside Decoder)
            pred_geo, pred_color, mask = decoder(batch)
            
            if mask is None:
                continue # value check
                
            # Ground Truths
            target_geo = batch['coord']  # We want to reconstruct original coordinates
            target_color = batch['feat'][:, :3] # Assuming first 3 feats are RGB (normalized)
            
            # Loss is calculated ONLY on MASKED points (The "holes")
            # pred_geo: [N, 3] -> we filter by mask [N]
            loss_g = criterion_geo(pred_geo[mask], target_geo[mask])
            loss_c = criterion_color(pred_color[mask], target_color[mask])
            
            # Total Loss (Weighted)
            loss = loss_g + (0.5 * loss_c)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_geo += loss_g.item()
            total_rgb += loss_c.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Geo: {total_geo/len(dataloader):.4f}, RGB: {total_rgb/len(dataloader):.4f})")
        
    # Save Model
    save_path = cfg.get('save_path', 'checkpoints/ptv3_inpaint.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(decoder.state_dict(), save_path)
    print(f"Saved Inpainting Model to {save_path}")

if __name__ == "__main__":
    train_inpaint_pipeline()
