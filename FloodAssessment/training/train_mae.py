import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob

from models.ptv3_encoder import SharedPTv3Encoder
from models.mae_decoder import MAEDecoder
from preprocessing.converter import prepare_data_for_ptv3

class PointDataset(Dataset):
    def __init__(self, root, voxel_size):
        all_files = glob.glob(os.path.join(root, "*.las")) + \
                    glob.glob(os.path.join(root, "*.laz")) + \
                    glob.glob(os.path.join(root, "*.ply"))
        
        # Filter out empty files
        self.files = [f for f in all_files if os.path.exists(f) and os.path.getsize(f) > 1024]
        
        if len(self.files) < len(all_files):
            print(f"Warning: Ignored {len(all_files) - len(self.files)} empty/small files.")
        self.voxel_size = voxel_size
        
        if len(self.files) == 0:
            print(f"Warning: No LAS/LAZ files found in {root}. Using dummy data for testing.")

    def __len__(self):
        return max(len(self.files), 1)

    def __getitem__(self, idx):
        if len(self.files) == 0:
            # Dummy data
            return {
                'coord': torch.randn(1000, 3),
                'feat': torch.randn(1000, 6),
                'grid_coord': torch.randint(0, 50, (1000, 3)),
                'offset': torch.IntTensor([1000]),
                'batch': torch.zeros(1000).long()
            }
            
        path = self.files[idx]
        try:
            data_dict, _ = prepare_data_for_ptv3(path, voxel_size=self.voxel_size)
            return data_dict
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

def collate_fn(batch):
    # Filter out None samples (failed loads)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # PTv3 requires specific collation (concatenating points, updating offsets/batch)
    # Simplified version:
    coords = []
    feats = []
    grid_coords = []
    offsets = []
    batch_idxs = []
    
    start_idx = 0
    for i, item in enumerate(batch):
        coord = item['coord']
        coords.append(coord)
        feats.append(item['feat'])
        grid_coords.append(item['grid_coord'])
        
        n = coord.shape[0]
        offsets.append(torch.IntTensor([start_idx + n]))
        batch_idxs.append(torch.full((n,), i, dtype=torch.long))
        start_idx += n
        
    return {
        'coord': torch.cat(coords),
        'feat': torch.cat(feats),
        'grid_coord': torch.cat(grid_coords),
        'offset': torch.cat(offsets),
        'batch': torch.cat(batch_idxs)
    }

def train_mae_pretraining():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)['mae']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Stage 0: MAE Pretraining on {device}")
    
    # Data
    dataset = PointDataset(cfg['data']['root'], cfg['data']['voxel_size'])
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0 # Win compatibility
    )
    
    # Model
    encoder = SharedPTv3Encoder()
    model = MAEDecoder(encoder, mask_ratio=cfg['model']['mask_ratio']).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    
    # Training Loop
    model.train()
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        for batch in dataloader:
            if batch is None:
                continue
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            reconstruction, mask = model(batch)
            
            # Loss Calculation (MSE on masked points)
            target = batch['feat'] # Reconstructing features
            
            if mask is not None:
                # Calculate loss only on masked points
                # mask is (N,) bool
                rec_loss = torch.nn.functional.mse_loss(reconstruction[mask], target[mask])
            else:
                rec_loss = torch.nn.functional.mse_loss(reconstruction, target)
                
            rec_loss.backward()
            optimizer.step()
            
            total_loss += rec_loss.item()
            
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Loss: {total_loss/len(dataloader):.4f}")
        
        # Save Checkpoint every epoch
        os.makedirs("checkpoints", exist_ok=True)
        # Save latest
        torch.save(model.encoder.state_dict(), cfg['training']['save_path'])
        # Save epoch specific (optional, can clutter disk if too many)
        torch.save(model.encoder.state_dict(), cfg['training']['save_path'].replace(".pth", f"_epoch_{epoch+1}.pth"))
        print(f"Saved checkpoint to {cfg['training']['save_path']}")
        
    # Save Encoder
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.encoder.state_dict(), cfg['training']['save_path'])
    print(f"Saved MAE Encoder to {cfg['training']['save_path']}")

if __name__ == "__main__":
    train_mae_pretraining()
