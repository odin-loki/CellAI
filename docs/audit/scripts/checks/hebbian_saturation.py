"""Section 8.2: the v1/v2 MetaplasticityLayer's positive-only Hebbian rule drives W to the all-ones matrix."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import torch
from cellai_core.memory import MetaplasticityLayer
torch.manual_seed(0)
D = 256
for scale, name in [(1 / D ** 0.5, "inputs N(0,1)/sqrt(D)"), (1.0, "inputs N(0,1)")]:
    L = MetaplasticityLayer(D); L.eval()
    X = torch.randn(20000, D) * scale
    with torch.no_grad():
        for t in range(20000):
            L(torch.randn(D) * 0.1, torch.randn(D) * 0.1, X[t])
            if t + 1 in (1000, 4000, 20000):
                W = L.W.data; s = torch.linalg.svdvals(W)
                print(f"{name:24s} tok={t+1:6d} frac(W>=0.999)={(W >= 0.999).float().mean():.3f} "
                      f"top-SV energy={(s[0]**2 / (s**2).sum()).item():.4f} |W|_F={W.norm():.1f}")
        outs = torch.stack([L.W.data @ torch.randn(D) for _ in range(50)])
        on = outs / outs.norm(dim=1, keepdim=True); cs = (on @ on.T).abs(); n = cs.shape[0]
        print("   mean |cos| between W x for 50 random x:", ((cs.sum() - n) / (n * n - n)).item())
