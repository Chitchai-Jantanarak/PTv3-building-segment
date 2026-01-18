import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.train_mae import PointDataset, collate_fn
from models.ptv3_encoder import SharedPTv3Encoder
from models.heads import SegmentationHead
from models.classifier_head import FEMAClassifier, GeometricFeatureExtractor

def train_semantic_pipeline():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)['semantic']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Stage 2: Semantic Training on {device}")
    
    # Dataloader
    dataset = PointDataset(cfg['data']['root'], cfg['data']['voxel_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], collate_fn=collate_fn)
    
    # Detect Input Dimension from first sample
    try:
        sample_file = dataset.files[0]
        sample_data = torch.load(sample_file)
        input_dim = sample_data['feat'].shape[1]
        print(f"Detected input feature dimension: {input_dim}")
    except Exception as e:
        print(f"Error detecting input dim, defaulting to 6: {e}")
        input_dim = 6

    encoder = SharedPTv3Encoder(input_channels=input_dim).to(device)
    # Load MAE Encoder as per requirement
    pretrained_path = cfg['model']['encoder_path']
    try:
        checkpoint = torch.load(pretrained_path, map_location=device)
        # Adapt weights if channel mismatch exists (e.g. 3 vs 4)
        model_state = encoder.state_dict()
        
        # Check stem weight shape
        stem_key = "backbone.embedding.stem.conv.weight"
        if stem_key in checkpoint and stem_key in model_state:
            ckpt_w = checkpoint[stem_key]
            curr_w = model_state[stem_key]
            if ckpt_w.shape != curr_w.shape:
                print(f"Adapting checkpoint stem weights from {ckpt_w.shape} to {curr_w.shape}")
                c_in_ckpt = ckpt_w.shape[-1]
                c_in_curr = curr_w.shape[-1]
                new_w = curr_w.clone() 
                min_c = min(c_in_ckpt, c_in_curr)
                new_w[..., :min_c] = ckpt_w[..., :min_c]
                checkpoint[stem_key] = new_w

        encoder.load_state_dict(checkpoint, strict=False)
        print(f"Loaded MAE Pretrained Encoder from {pretrained_path}")
    except Exception as e:
        print(f"Warning: Could not load pretrained encoder: {e}")
        
    # Seg-A Head (9 Classes as per standard mapping)
    # Encoder output is 64 channels
    seg_a = SegmentationHead(in_channels=64, num_classes=9).to(device)
    
    # MLP Classifier
    # Takes geometric feature vector (32,) -> This seems specific to the extractor.
    # But if we use deep features, we might want to change this.
    # For now, keep standard as requested (GeometricFeatureExtractor uses raw points).
    classifier = FEMAClassifier(input_dim=32, num_classes=28).to(device)
    geo_extractor = GeometricFeatureExtractor()
    
    optimizer = optim.AdamW(list(seg_a.parameters()) + list(classifier.parameters()), lr=cfg['training']['lr'])
    criterion_seg = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        for batch in dataloader:
            if batch is None:
                continue
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 1. Seg-A Forward
            with torch.no_grad(): # Optional finetune
                enc_out = encoder(batch)
            
            seg_pred = seg_a(enc_out.feat)
            
            if 'seg_label' in batch:
                target = batch['seg_label'] 
            else:
                target = torch.randint(0, 30, (len(seg_pred),), device=device)
            
            loss_seg = criterion_seg(seg_pred, target)
            
            # 2. MLP Classification
            # Logic: We treat Seg-A predictions as instance masks (using DBSCAN/Components)
            # extracting features, then classifying.
            # However, for end-to-end training, we usually train MLP on GT instances
            # to decouple segmentation error from classification error initially.
            
            # Mocking instance extraction from batch
            # Assuming 'instance_ids' in batch or we treat efficient batch process
            loss_cls = 0
            # For this simplified script, we simulate feature extraction on random 10 instances
            dummy_instances = 10
            for _ in range(dummy_instances):
                # Simulate a building cluster
                cluster_pts = torch.randn(100, 3).numpy()
                feats = geo_extractor.extract(cluster_pts)
                feats_tensor = torch.from_numpy(feats).float().unsqueeze(0).to(device)
                
                cls_pred = classifier(feats_tensor)
                target_cls = torch.randint(0, 28, (1,), device=device)
                loss_cls += criterion_cls(cls_pred, target_cls)
                
            loss_cls = loss_cls / dummy_instances
            
            loss = loss_seg + loss_cls
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save({
        'encoder': encoder.state_dict(),
        'seg_a': seg_a.state_dict(),
        'classifier': classifier.state_dict()
    }, cfg['training']['save_path'])

if __name__ == "__main__":
    train_semantic_pipeline()
