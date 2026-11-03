import sys, time, statistics, resource, json, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import torch
torch.set_num_threads(int(os.environ.get("NT", "4")))
from cellai_core.base import ModelParams
from v1.cell_ai import CellAI
from v2.cell_ai_v2 import CellAIv2
from v3.cell_ai_v3 import CellAIv3, V3Config
OUT = "" + os.environ.get('AUDIT_OUT', 'audit_out') + ""

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def timeit(fn, reps=50, warmup=5):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(ts), "p10_ms": sorted(ts)[len(ts)//10], "p90_ms": sorted(ts)[int(len(ts)*0.9)-1]}

def build(name):
    """Named configs."""
    if name == "v1_default":   return CellAI(ModelParams(device="cpu"))
    if name == "v2_default":   return CellAIv2(ModelParams(device="cpu"))
    if name == "v3_default":   return CellAIv3(ModelParams(device="cpu"), V3Config())
    if name == "v1_small":     return CellAI(ModelParams(device="cpu", state_size=64, num_partitions=2))
    if name == "v2_small":     return CellAIv2(ModelParams(device="cpu", state_size=64, num_partitions=2))
    if name == "v3_small":
        return CellAIv3(ModelParams(device="cpu", state_size=64, num_partitions=2),
                        V3Config(pde_type="spectral", hebbian_type="sparse", partition_type="multiscale",
                                 N_fast=2, D_fast=32, N_slow=2, D_slow=64, K_slow=8, label="v3_small"))
    if name == "v3_E20":
        # True E20 (arch_search/run_arch_search_v3_resume.py:122; confirmed by results_v3.json "info"):
        # spectral + sparse Hebbian + multiscale, no resonance, 53,512,480 params.
        return CellAIv3(ModelParams(device="cpu", state_size=512, num_partitions=8),
                        V3Config(label="E20", pde_type="spectral", hebbian_type="sparse", partition_type="multiscale",
                                 resonance_type="none", N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8))
    if name == "v3_E14":
        # E14 / superseded run_arch_search_v3.py "E20": full Hebbian + per-frequency resonance.
        return CellAIv3(ModelParams(device="cpu", state_size=512, num_partitions=8),
                        V3Config(label="E14", pde_type="spectral", hebbian_type="full", partition_type="multiscale",
                                 resonance_type="per_freq", N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8))
    if name == "v3_E34":
        return CellAIv3(ModelParams(device="cpu", state_size=1024, num_partitions=8),
                        V3Config(label="E34", pde_type="spectral", hebbian_type="sparse", partition_type="multiscale",
                                 resonance_type="none", N_fast=8, D_fast=256, N_slow=4, D_slow=512, K_slow=8,
                                 k_frac=0.125, use_bypass=True))
    raise KeyError(name)

def step_fn(model):
    """Per-token step for any version: v1/v2 have cellular_step too."""
    return model.cellular_step
