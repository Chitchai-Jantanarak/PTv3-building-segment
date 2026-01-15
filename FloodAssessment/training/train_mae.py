import sys
import os

# Force spconv to use simple algorithms (prevents crash on RTX 5090)
# Force spconv to use simple algorithms (prevents crash on RTX 5090)
os.environ["SPCONV_ALGO"] = "native"
# Mitigate OOM fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import cumm # Critical: Load bindings before spconv
import yaml
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob

# Add project root to sys.path to find 'models' and 'preprocessing'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ptv3_encoder import SharedPTv3Encoder
from models.mae_decoder import MAEDecoder
from preprocessing.converter import prepare_data_for_ptv3

class PointDataset(Dataset):
    def __init__(self, root, voxel_size, max_points=None):
        all_files = glob.glob(os.path.join(root, "*.las")) + \
                    glob.glob(os.path.join(root, "*.laz")) + \
                    glob.glob(os.path.join(root, "*.ply")) + \
                    glob.glob(os.path.join(root, "*.pth"))
        
        # Filter out empty files
        self.files = [f for f in all_files if os.path.exists(f) and os.path.getsize(f) > 1024]
        
        if len(self.files) < len(all_files):
            print(f"Warning: Ignored {len(all_files) - len(self.files)} empty/small files.")
        self.voxel_size = voxel_size
        self.max_points = max_points
        
        if len(self.files) == 0:
            print(f"Warning: No LAS/LAZ/PTH files found in {root}. Using dummy data for testing.")

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
            if path.endswith('.pth'):
                data_dict = torch.load(path)
                # Ensure compatibility if max_points is set (though preprocessing usually handles it)
                # But if we want to enforce it on .pth files too:
                # Sanity check for corruption (NaNs/Infs) that causes Core Dump
                if torch.isnan(data_dict['coord']).any() or torch.isinf(data_dict['coord']).any() or \
                   torch.isnan(data_dict['feat']).any() or torch.isinf(data_dict['feat']).any():
                    print(f"Warning: Corrupt data (NaN/Inf) found in {path}. Skipping.")
                    return None
                    
                # Ensure compatibility if max_points is set
                if self.max_points and len(data_dict['coord']) > self.max_points:
                     # Simple random downsample
                     choice = torch.randperm(len(data_dict['coord']))[:self.max_points]
                     data_dict['coord'] = data_dict['coord'][choice]
                     data_dict['feat'] = data_dict['feat'][choice]
                     data_dict['grid_coord'] = data_dict['grid_coord'][choice]
                     data_dict['offset'] = torch.IntTensor([len(data_dict['coord'])])
                     data_dict['batch'] = torch.zeros(len(data_dict['coord'])).long()
                return data_dict
            else:
                data_dict, _ = prepare_data_for_ptv3(path, voxel_size=self.voxel_size, max_points=self.max_points)
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
    dataset = PointDataset(
        cfg['data']['root'], 
        cfg['data']['voxel_size'],
        max_points=cfg['data'].get('max_points', 80000)
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0 # Win compatibility
    )
    
    # Model
    # Determine input channels from dataset sample
    sample_data = dataset[0]
    input_dim = sample_data['feat'].shape[1]
    print(f"Detected input feature dimension: {input_dim}")
    
    encoder = SharedPTv3Encoder(input_channels=input_dim)
    model = MAEDecoder(encoder, mask_ratio=cfg['model']['mask_ratio'], enc_channels=64, target_channels=input_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    
    best_loss = float('inf')
    
    # Resume from checkpoint if exists
    start_epoch = 0
    save_path = cfg['training']['save_path']
    # If the user-defined save path exists (meaning we have a 'latest' checkpoint), load it
    if os.path.exists(save_path):
        print(f"Resuming from checkpoint: {save_path}")
        try:
            # Check if it's a full model checkpoint or just encoder
            checkpoint = torch.load(save_path, map_location=device)
            # If it's a state dict, try loading it
            if isinstance(checkpoint, dict):
                # Try loading into encoder first (since we ignore decoder weights usually)
                try:
                    model.encoder.load_state_dict(checkpoint, strict=False)
                    print("Loaded encoder weights successfully.")
                except Exception as e:
                    print(f"Could not load as encoder weights: {e}")
            else:
                print("Checkpoint format unrecognized, starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    else:
        print("No existing checkpoint found. Starting fresh.")

    # Training Loop
    model.train()
    print(f"Detailed logs will be printed every 10 batches.")
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Aggressive cache clearing
            torch.cuda.empty_cache()
            
            if batch is None:
                continue
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            try:
                reconstruction, mask = model(batch)
            except AttributeError as e:
                 print(f"Forward pass error: {e}. Ensure PointTransformerV3 is returning a Point/Dict object.")
                 raise e
            
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
            
            loss_val = rec_loss.item()
            total_loss += loss_val
            
            # Verbose logging
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}] Batch [{batch_idx+1}/{num_batches}] Loss: {loss_val:.4f}")
            
        avg_loss = total_loss / max(1, len(dataloader))
        print(f"==> Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        
        # Save latest (Overwrite previous to save space)
        torch.save(model.encoder.state_dict(), save_path)
        print(f"Updated Latest Checkpoint: {os.path.abspath(save_path)}")
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_save_path = save_path.replace(".pth", "_best.pth")
            torch.save(model.encoder.state_dict(), best_save_path)
            print(f"New Best Model! Saved to: {os.path.abspath(best_save_path)}")
        
    # Save Encoder
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.encoder.state_dict(), cfg['training']['save_path'])
    print(f"Saved MAE Encoder to {cfg['training']['save_path']}")

if __name__ == "__main__":
    train_mae_pretraining()
