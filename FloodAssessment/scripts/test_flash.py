try:
    import flash_attn
    print(f"Flash Attention version: {flash_attn.__version__}")
    
    import torch
    q = torch.randn(1, 10, 32, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 10, 32, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 10, 32, 64, device='cuda', dtype=torch.float16)
    
    from flash_attn import flash_attn_func
    out = flash_attn_func(q, k, v)
    print("Flash Attention functional test passed!")
except ImportError:
    print("Flash Attention NOT installed.")
except Exception as e:
    print(f"Flash Attention FAILED: {e}")
