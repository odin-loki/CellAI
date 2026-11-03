"""Parameter census of v1, v2, v3-default, E20, E24/E27 and E34 (paper section 5.1)."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import torch
torch.set_num_threads(4)
from cellai_core.base import ModelParams
from v1.cell_ai import CellAI
from v2.cell_ai_v2 import CellAIv2
from v3.cell_ai_v3 import CellAIv3, V3Config
def count(m):
    tot = sum(p.numel() for p in m.parameters()); emb = m.encoder.embedding.weight.numel()
    return tot, tot-emb, emb/tot
cfgs = {
 "v1": lambda: CellAI(ModelParams(device="cpu")),
 "v2": lambda: CellAIv2(ModelParams(device="cpu")),
 "v3def": lambda: CellAIv3(ModelParams(device="cpu"), V3Config()),
 "E20": lambda: CellAIv3(ModelParams(device="cpu", state_size=512, num_partitions=8), V3Config(pde_type="spectral", hebbian_type="sparse", partition_type="multiscale", resonance_type="none", N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8, k_frac=0.125)),
 "E24/E27": lambda: CellAIv3(ModelParams(device="cpu", state_size=1024, num_partitions=8), V3Config(pde_type="spectral", hebbian_type="sparse", partition_type="multiscale", resonance_type="none", N_fast=8, D_fast=256, N_slow=4, D_slow=512, K_slow=8, k_frac=0.125)),
 "E34": lambda: CellAIv3(ModelParams(device="cpu", state_size=1024, num_partitions=8), V3Config(pde_type="spectral", hebbian_type="sparse", partition_type="multiscale", resonance_type="none", N_fast=8, D_fast=256, N_slow=4, D_slow=512, K_slow=8, k_frac=0.125, use_bypass=True)),
}
for k,f in cfgs.items():
    m=f(); t,ne,sh=count(m); print(k, f"{t:,}", f"{ne:,}", f"{sh*100:.1f}%")
    if k=="E20":
        for n,mod in m.partitions.named_children():
            print("  ",n, sum(p.numel() for p in mod.parameters()))
        print("  slow pde", sum(p.numel() for p in m.partitions.slow_pm.pde.parameters()), "input_gate", m.partitions.slow_pm.pde.input_gate.weight.numel())
        print("  fast pde", sum(p.numel() for p in m.partitions.fast_pm.pde.parameters()))
