"""
Full Cell AI test + benchmark after rewrite.
Run (from repo root):  .venv\Scripts\python scripts\_test_full.py > C:\Temp\results.log
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

def bench(label, fn, reps=50, warmup=5):
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
    print(f"  {label:<52} {elapsed:7.2f} ms/call", flush=True)
    return elapsed

# ── GPU status ──────────────────────────────────────────────────────────────
print("=" * 65)
print("GPU STATUS")
print("=" * 65)
import cupy as cp
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"CuPy    : {cp.__version__} (available but not used in partition step)")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Core components ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("CORE COMPONENTS")
print("=" * 65)

from cellai_core.base    import ModelParams, CellularPDE
from cellai_core.encoder import UniversalEncoder
from cellai_core.memory  import MemoryFormation
from cellai_core.partition import PartitionManager

params = ModelParams(state_size=256, num_partitions=4)
enc = UniversalEncoder(256).to(DEVICE)
TEXT = "The quick brown fox jumps over the lazy dog." * 5

bench("UniversalEncoder.encode_pooled",   lambda: enc.encode_pooled(TEXT, device=DEVICE))

pde = CellularPDE(4, 256).to(DEVICE)
state = torch.zeros(4, 256, device=DEVICE)
inp   = torch.randn(256, device=DEVICE)
bench("CellularPDE.step (4×256, batched)", lambda: pde.step(state, inp, params))

pm = PartitionManager(params)
bench("PartitionManager.step",            lambda: pm.step(inp))
del pm; gc.collect()
if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

mf = MemoryFormation(256, 100).to(DEVICE)
bench("MemoryFormation.forward",          lambda: mf(inp, inp))

# ── Version tests ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("VERSION TESTS  (v1 and v2 only — v3 removed)")
print("=" * 65)
from v1.cell_ai    import CellAI
from v2.cell_ai_v2 import CellAIv2

for VersionClass in [CellAI, CellAIv2]:
    m = VersionClass(params)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m.forward("hello world")
    elapsed = ms(t0)
    print(f"  {m.VERSION}: device={m.device}  output={tuple(out.shape)}  {elapsed:.1f} ms", flush=True)
    bench(f"  {m.VERSION} forward (50 reps)", lambda m=m: m.forward("hello world"))
    del out, m; gc.collect()
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

print("Version tests PASSED")

# ── Model tests ──────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MODEL TESTS  (v1 and v2 × 8 models = 16 combinations)")
print("=" * 65)

from models.registry import get_model, list_models, list_versions

passed = 0
failed = 0
timings = []
for version in list_versions():
    for mtype in list_models():
        t0 = time.perf_counter()
        try:
            m   = get_model(mtype, version=version, params=params)
            r   = m.chat("What is artificial intelligence?")
            elapsed = int(ms(t0))
            print(f"  PASS  {version}/{mtype:<16}  {elapsed:>6}ms  len={len(r)}", flush=True)
            timings.append(elapsed)
            passed += 1
        except BaseException as e:
            import traceback
            elapsed = int(ms(t0))
            print(f"  FAIL  {version}/{mtype:<16}  {elapsed:>6}ms  {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed += 1
        finally:
            try:
                del m
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

print(f"\nModels: {passed} PASSED, {failed} FAILED")
if timings:
    print(f"Median latency: {sorted(timings)[len(timings)//2]} ms")

# ── Scaling ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SCALING  (state_size vs. throughput, v1)")
print("=" * 65)
for sz in [64, 128, 256, 512, 1024]:
    p = ModelParams(state_size=sz, num_partitions=4)
    m = CellAI(p)
    bench(f"state_size={sz}", lambda m=m: m.forward("hello"), reps=20, warmup=3)
    del m; gc.collect()
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()

# ── Math pipeline ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MATH PIPELINE")
print("=" * 65)
import tempfile, json
from pathlib import Path
from data.pipelines.math_pipeline import generate

with tempfile.TemporaryDirectory() as tmpdir:
    counts = generate(count=50, output_dir=Path(tmpdir), seed=42)
    lines  = (Path(tmpdir) / "train.jsonl").read_text(encoding="utf-8").splitlines()
    sample = json.loads(lines[0])
    print(f"Generated: {counts}  train_samples={len(lines)}")
    print(f"Sample: [{sample['domain']}/{sample['difficulty']}] {sample['problem'][:60]}")

print("Math pipeline PASSED")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"Models:   {passed}/{passed+failed} passed")
print(f"GPU:      {torch.cuda.is_available()}")
if failed == 0:
    print("\nALL TESTS PASSED")
else:
    print(f"\n{failed} TESTS FAILED")
