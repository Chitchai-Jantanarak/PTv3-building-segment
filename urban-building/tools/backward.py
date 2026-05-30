import os
import sys

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

    out = conv(x)
    loss = out.features.sum()
    loss.backward()
    torch.cuda.synchronize()
    print(f"[OK] {algo_name} strided={strided}: grad_norm={feat.grad.norm().item():.4f}", flush=True)


if __name__ == "__main__":
    print("torch", torch.__version__, "cap", torch.cuda.get_device_capability())
    print("spconv", spconv.__version__, spconv.__file__)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cases = {
        "subm_native": ("Native", ConvAlgo.Native, False),
        "subm_gemm": ("MaskImplicitGemm", ConvAlgo.MaskImplicitGemm, False),
        "strided_native": ("Native", ConvAlgo.Native, True),
        "strided_gemm": ("MaskImplicitGemm", ConvAlgo.MaskImplicitGemm, True),
    }
    if which == "all":
        for name, (a, alg, st) in cases.items():
            run(a, alg, st)
    else:
        a, alg, st = cases[which]
        run(a, alg, st)
