"""
Baseline profiler — measures every hot path in the current Cell AI code.
Run (from repo root): .venv\Scripts\python scripts\_profile_baseline.py
"""
import gc
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np

torch.set_grad_enabled(False)

def ms(t0): return (time.perf_counter() - t0) * 1000

# ─── Helpers ────────────────────────────────────────────────────────────────
def bench(name, fn, reps=50, warmup=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / reps * 1000
    print(f"  {name:<48} {elapsed:8.2f} ms/call", flush=True)
    return elapsed

# ─── Setup ──────────────────────────────────────────────────────────────────
from cellai_core.base import ModelParams, MemoryCellBase
from cellai_core.encoder import UniversalEncoder
from cellai_core.memory import MemoryFormation, MetaplasticityLayer
from cellai_core.partition import PartitionManager, _CUPY_AVAILABLE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"CuPy GPU: {_CUPY_AVAILABLE}")
print()

# ─── 1. Tokenizer / Encoder ─────────────────────────────────────────────────
print("=" * 65)
print("1. ENCODER")
print("=" * 65)
enc = UniversalEncoder(256).to(DEVICE)
TEXT = "The quick brown fox jumps over the lazy dog." * 5

bench("tokenize (no embed)", lambda: enc._tokenizer.encode(TEXT))
bench("encode_pooled (256-dim)",   lambda: enc.encode_pooled(TEXT, device=DEVICE))
bench("encode (seq output)",       lambda: enc.encode(TEXT, device=DEVICE))

# ─── 2. MemoryCellBase ──────────────────────────────────────────────────────
print()
print("=" * 65)
print("2. MEMORY CELL (single PDE step)")
print("=" * 65)
params = ModelParams(state_size=256, num_partitions=4)
cell = MemoryCellBase(256).to(DEVICE)
state  = torch.randn(256, device=DEVICE)
inp    = torch.randn(256, device=DEVICE)
nbrs   = torch.randn(2, 256, device=DEVICE)

bench("MemoryCellBase.forward",     lambda: cell(state, inp, nbrs, params))
bench("f_term only",                lambda: cell.f_term(state, inp))
bench("diffusion only",             lambda: cell.diffusion_term(state, nbrs, 0.1))

# ─── 3. MemoryFormation ─────────────────────────────────────────────────────
print()
print("=" * 65)
print("3. MEMORY FORMATION")
print("=" * 65)
mf = MemoryFormation(256, 100).to(DEVICE)
bench("MemoryFormation.forward",    lambda: mf(inp, state))

# ─── 4. Partition step ──────────────────────────────────────────────────────
print()
print("=" * 65)
print("4. PARTITION MANAGER (4 partitions, 256-dim)")
print("=" * 65)
pm4 = PartitionManager(params)
inputs4 = [np.random.randn(256).astype(np.float32) for _ in range(4)]
bench("PartitionManager.step (4×256)", lambda: pm4.step(inputs4))
pm4.shutdown(); del pm4; gc.collect()
if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

# larger partition
params8 = ModelParams(state_size=256, num_partitions=8)
pm8 = PartitionManager(params8)
inputs8 = [np.random.randn(256).astype(np.float32) for _ in range(8)]
bench("PartitionManager.step (8×256)", lambda: pm8.step(inputs8))
pm8.shutdown(); del pm8; gc.collect()
if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

# ─── 5. GPU↔CPU transfer overhead ──────────────────────────────────────────
print()
print("=" * 65)
print("5. GPU<->CPU TRANSFER (overhead in process_state)")
print("=" * 65)
t_gpu = torch.randn(256, device=DEVICE)
bench("tensor.cpu().numpy() [256]", lambda: t_gpu.cpu().numpy())
bench("tensor.cpu().numpy() [1024]",lambda: torch.randn(1024, device=DEVICE).cpu().numpy())
bench("torch.tensor(np_arr).cuda", lambda: torch.tensor(np.random.randn(256).astype(np.float32), device=DEVICE))

# ─── 6. Full v1 forward ─────────────────────────────────────────────────────
print()
print("=" * 65)
print("6. FULL v1 CellAI forward (state_size=256, 4 partitions)")
print("=" * 65)
from v1.cell_ai import CellAI
m = CellAI(params)
bench("CellAI.forward('hello')",    lambda: m.forward("hello world"))
bench("CellAI.forward(long text)",  lambda: m.forward(TEXT))
del m; gc.collect()
if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

# ─── 7. Full v2 forward (includes CrystalLattice Python loops) ──────────────
print()
print("=" * 65)
print("7. FULL v2 CellAIv2 forward (state_size=256, 4 partitions)")
print("=" * 65)
from v2.cell_ai_v2 import CellAIv2, CrystalLattice, ResonanceSystem
m2 = CellAIv2(params)
bench("CellAIv2.forward('hello')",  lambda: m2.forward("hello world"))

# Profile the specific v2 pieces
res = ResonanceSystem(256, DEVICE)
latt = CrystalLattice(256, device=DEVICE)
st = torch.randn(256, device=DEVICE)
bench("ResonanceSystem.compute_resonance", lambda: res.compute_resonance(st))
bench("ResonanceSystem.wave_interaction",  lambda: res.wave_interaction(st))
bench("CrystalLattice.forward (Python loops!)", lambda: latt(st), reps=10, warmup=2)
del m2, res, latt; gc.collect()
if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

# ─── 8. Domain model forward ────────────────────────────────────────────────
print()
print("=" * 65)
print("8. DOMAIN MODELS (v1 backbone, state_size=256)")
print("=" * 65)
from models.registry import get_model
for mtype in ['nlp', 'math', 'software', 'cot', 'multimodal']:
    m = get_model(mtype, 'v1', params)
    bench(f"{mtype}.chat('hello')",  lambda m=m: m.chat("hello world"), reps=20)
    del m; gc.collect()
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

# ─── 9. Scaling: state_size impact ──────────────────────────────────────────
print()
print("=" * 65)
print("9. SCALING — state_size vs. throughput")
print("=" * 65)
for sz in [64, 128, 256, 512, 1024]:
    p = ModelParams(state_size=sz, num_partitions=4)
    m = CellAI(p)
    t = bench(f"CellAI.forward (state_size={sz:<4})", lambda m=m: m.forward("hello world"), reps=20)
    del m; gc.collect()
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

print()
print("Profiling COMPLETE.")
