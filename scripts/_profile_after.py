"""
Post-rewrite profiler. Run (from repo root): .venv\Scripts\python scripts\_profile_after.py
"""
import gc
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bench(name, fn, reps=100, warmup=10):
    for _ in range(warmup): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {name:<52} {ms:7.2f} ms")
    return ms

from cellai_core.base    import ModelParams, CellularPDE
from cellai_core.encoder import UniversalEncoder
from cellai_core.memory  import MemoryFormation
from cellai_core.partition import PartitionManager
from v1.cell_ai    import CellAI
from v2.cell_ai_v2 import CellAIv2, CrystalLattice, ResonanceSystem
from models.registry import get_model

params = ModelParams(state_size=256, num_partitions=4)
TEXT = "The quick brown fox jumps over the lazy dog." * 5

print("=" * 65)
print("POST-REWRITE PERFORMANCE (all ms/call, 100 reps)")
print("=" * 65)

enc = UniversalEncoder(256).to(DEVICE)
bench("UniversalEncoder.encode_pooled", lambda: enc.encode_pooled(TEXT, device=DEVICE))

pde = CellularPDE(4, 256).to(DEVICE)
s = torch.zeros(4, 256, device=DEVICE)
i = torch.randn(256, device=DEVICE)
bench("CellularPDE.step (4×256, batched GPU)", lambda: pde.step(s, i, params))

pm = PartitionManager(params)
bench("PartitionManager.step",  lambda: pm.step(i))
del pm; gc.collect()

mf = MemoryFormation(256, 100).to(DEVICE)
bench("MemoryFormation.forward", lambda: mf(i, i))

m1 = CellAI(params)
bench("v1 CellAI.forward",      lambda: m1.forward("hello world"))
del m1; gc.collect()
if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

m2 = CellAIv2(params)
bench("v2 CellAIv2.forward",    lambda: m2.forward("hello world"))

res = ResonanceSystem(256).to(DEVICE)
bench("ResonanceSystem.forward", lambda: res(i))

lat = CrystalLattice(256).to(DEVICE)
bench("CrystalLattice.forward (vectorised einsum)", lambda: lat(i), reps=200, warmup=20)

del m2, res, lat; gc.collect()
if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

print()
print("Domain models (steady-state, 50 reps):")
for version in ['v1', 'v2']:
    for mtype in ['nlp', 'math', 'software', 'cot', 'multimodal']:
        m = get_model(mtype, version=version, params=params)
        bench(f"  {version}/{mtype}.chat", lambda m=m: m.chat("hello world"), reps=50, warmup=5)
        del m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()
