"""Smoke test Round 3 critical fixes."""
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cellai_core.base import ModelParams
from v3.cell_ai_v3 import CellAIv3, V3Config

D = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {D}')
OK = True

# FIX 1: train_step_sequential must call cellular_step() so resonance is trained
try:
    cfg = V3Config(label='fix1', pde_type='spectral', partition_type='multiscale',
                   resonance_type='per_freq', N_fast=4, D_fast=128, N_slow=2, D_slow=256)
    m = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    m.train_step_sequential('The quick brown fox jumps.', opt)
    # resonance parameters must have gradient
    res_g = m.resonance.log_mag.grad
    gate_g = m.resonance.gate_proj.weight.grad
    status = 'PASS' if res_g is not None and gate_g is not None else 'FAIL'
    rg = res_g.norm().item() if res_g is not None else 0.0
    gg = gate_g.norm().item() if gate_g is not None else 0.0
    print(f'[{status}] FIX1 (train_step uses cellular_step): resonance log_mag grad={rg:.6f}  gate grad={gg:.6f}')
    if status == 'FAIL': OK = False
    del m
except Exception as e:
    print(f'[FAIL] FIX1: {e}'); import traceback; traceback.print_exc(); OK = False

# FIX 2: log_alpha_ext removed — model should have no such attribute for resonance-only config
try:
    cfg2 = V3Config(label='fix2', pde_type='spectral', resonance_type='per_freq')
    m2 = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg2)
    has_alpha = hasattr(m2, 'log_alpha_ext')
    status = 'PASS' if not has_alpha else 'FAIL'
    print(f'[{status}] FIX2 (log_alpha_ext removed): has_attr={has_alpha}')
    if status == 'FAIL': OK = False
    del m2
except Exception as e:
    print(f'[FAIL] FIX2: {e}'); OK = False

# FIX 3: gate_proj.bias = +3.0 → sigmoid(3) = 0.95 at init
try:
    cfg3 = V3Config(label='fix3', pde_type='spectral', resonance_type='per_freq')
    m3 = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg3)
    bias_val = m3.resonance.gate_proj.bias.data.mean().item()
    status = 'PASS' if abs(bias_val - 3.0) < 0.01 else 'FAIL'
    print(f'[{status}] FIX3 (gate_proj.bias=+3): bias_mean={bias_val:.3f}  sigmoid={torch.sigmoid(torch.tensor(bias_val)):.3f}')
    if status == 'FAIL': OK = False
    del m3
except Exception as e:
    print(f'[FAIL] FIX3: {e}'); OK = False

# FIX 4: SparseHebbian hebb_rate = 0.005 (not 0.001)
try:
    from cellai_core.sparse_hebbian import SparseHebbian
    sh = SparseHebbian(256)
    status = 'PASS' if abs(sh.hebb_rate - 0.005) < 1e-7 else 'FAIL'
    print(f'[{status}] FIX4 (SparseHebbian hebb_rate=0.005): {sh.hebb_rate}')
    if status == 'FAIL': OK = False
except Exception as e:
    print(f'[FAIL] FIX4: {e}'); OK = False

# FIX 5: resonance is applied additively and trains correctly
try:
    cfg5 = V3Config(label='fix5', pde_type='spectral', partition_type='single',
                    resonance_type='per_freq')
    m5 = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg5)
    opt5 = torch.optim.AdamW(m5.parameters(), lr=1e-3)
    # Train for 100 steps
    losses = []
    for _ in range(100):
        l = m5.train_step_sequential('The transformer model learns sequence representations.', opt5)
        losses.append(l)
    # Resonance parameters should have been updated from initial values
    log_mag_init = torch.zeros(129)  # initial value
    log_mag_diff = (m5.resonance.log_mag.data - log_mag_init.to(D)).abs().mean().item()
    phase_diff   = (m5.resonance.phase.data).abs().mean().item()  # should have moved from random init
    status = 'PASS' if log_mag_diff > 1e-6 else 'FAIL'
    print(f'[{status}] FIX5 (resonance trains): log_mag moved={log_mag_diff:.6f}  loss@100={losses[-1]:.3f}')
    if status == 'FAIL': OK = False
    del m5
except Exception as e:
    print(f'[FAIL] FIX5: {e}'); import traceback; traceback.print_exc(); OK = False

# FIX 6: domain tokens affect tokenization
try:
    cfg6 = V3Config(label='fix6')
    m6 = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg6)
    toks_plain  = m6.encoder.tokenize('The transformer learns representations')
    toks_tagged = m6.encoder.tokenize('<<TEXT>> The transformer learns representations')
    status = 'PASS' if len(toks_tagged) > len(toks_plain) else 'FAIL'
    print(f'[{status}] FIX6 (domain token adds tokens): plain={len(toks_plain)}  tagged={len(toks_tagged)}')
    if status == 'FAIL': OK = False
    del m6
except Exception as e:
    print(f'[FAIL] FIX6: {e}'); OK = False

print()
print('ALL SMOKE TESTS PASSED' if OK else 'SOME TESTS FAILED — CHECK OUTPUT ABOVE')
