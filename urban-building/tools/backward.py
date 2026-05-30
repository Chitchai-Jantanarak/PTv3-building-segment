import os

os.environ.setdefault("SPCONV_ALGO", "native")

import torch
import spconv
import spconv.pytorch as spconv_pt
from spconv.core import ConvAlgo


def run(algo_name, algo, strided):
    torch.manual_seed(0)
    device = torch.device("cuda")
    n = 4096
    coord = torch.randint(0, 128, (n, 3), dtype=torch.int32, device=device)
    batch_idx = torch.zeros(n, 1, dtype=torch.int32, device=device)
    indices = torch.cat([batch_idx, coord], dim=1)
    feat = torch.randn(n, 16, device=device, requires_grad=True)

    x = spconv_pt.SparseConvTensor(feat, indices, [128, 128, 128], 1)

    if strided:
        conv = spconv_pt.SparseConv3d(16, 32, 3, stride=2, bias=False, algo=algo).to(device)
    else:
        conv = spconv_pt.SubMConv3d(16, 32, 3, bias=False, algo=algo).to(device)

    try:
        out = conv(x)
        loss = out.features.sum()
        loss.backward()
        torch.cuda.synchronize()
        print(f"[OK] {algo_name} strided={strided}: grad_norm={feat.grad.norm().item():.4f}")
    except Exception as e:
        print(f"[FAIL] {algo_name} strided={strided}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("torch", torch.__version__, "cap", torch.cuda.get_device_capability())
    print("spconv", spconv.__version__, spconv.__file__)
    for strided in (False, True):
        run("Native", ConvAlgo.Native, strided)
        run("MaskImplicitGemm", ConvAlgo.MaskImplicitGemm, strided)
