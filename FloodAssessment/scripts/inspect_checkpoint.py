import torch
import os
import sys

def inspect_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        # Map location to CPU to avoid CUDA errors if just inspecting
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint type: Dict with keys: {list(checkpoint.keys())}")
            
            # Identify state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif any(k.startswith('encoder') or k.startswith('backbone') for k in checkpoint.keys()):
                 # The dict itself might be the state dict
                 state_dict = checkpoint
            else:
                print("Could not identify state_dict. Printing top-level keys only.")
                return

            print(f"\nTotal parameters: {len(state_dict)}")
            print("-" * 50)
            
            # Print hierarchy summary
            # We'll group by top-level modules to make it readable
            modules = {}
            for key in state_dict.keys():
                parts = key.split('.')
                top_level = parts[0]
                if top_level not in modules:
                    modules[top_level] = []
                modules[top_level].append(key)
            
            for mod, keys in modules.items():
                print(f"Module: {mod} ({len(keys)} parameters/buffers)")
                # Print first few to show structure
                for k in keys[:3]:
                    print(f"  - {k}   {state_dict[k].shape}")
                if len(keys) > 3:
                     print(f"  ... (+ {len(keys)-3} more)")
                print("")

        else:
            print("Checkpoint is not a dictionary (maybe just model object?)")

    except Exception as e:
        print(f"Failed to inspect checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default path based on previous context
        path = r"e:\Lesson\2568-2\040613703_aidev\DoingSomeWeirdLikeWeirdo\FloodAssessment\checkpoints\ptv3_mae_encoder_best.pth"
    
    inspect_checkpoint(path)
