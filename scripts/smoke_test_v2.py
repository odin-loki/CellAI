"""Smoke test all v2 fixes."""
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

# Test 1: SpectralPDE + MultiScale (E8)
try:
    cfg = V3Config(label='smoke_E8', pde_type='spectral', partition_type='multiscale',
                   resonance_type='none', hebbian_type='full',
                   N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8)
    m = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg)
    out = m.forward('hello world test')
    print(f'[PASS] E8 (Spectral+MultiScale): shape={out.shape}, norm={out.norm():.4f}')
    del m
except Exception as e:
    print(f'[FAIL] E8: {e}'); OK = False

# Test 2: Fixed PerFreqResonance (multiplicative gate) - gradient must be non-zero
try:
    cfg2 = V3Config(label='smoke_E9', pde_type='spectral', resonance_type='per_freq')
    m2 = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg2)
    out2 = m2.forward('hello world test')
    out2.sum().backward()
    gate_grad = m2.resonance.gate_proj.weight.grad.norm().item()
    phase_grad = m2.resonance.phase.grad.norm().item()
    status = 'PASS' if gate_grad > 0 and phase_grad > 0 else 'FAIL'
    print(f'[{status}] E9 (PerFreq gate): gate_grad={gate_grad:.6f}, phase_grad={phase_grad:.6f}')
    if status == 'FAIL': OK = False
    del m2
except Exception as e:
    print(f'[FAIL] E9: {e}'); OK = False

# Test 3: SparseHebbian deferred (no interference)
try:
    cfg3 = V3Config(label='smoke_E10', pde_type='spectral', hebbian_type='sparse', k_frac=0.125)
    m3 = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg3)
    opt = torch.optim.AdamW(m3.parameters(), lr=3e-4)
    loss_val = m3.train_step_sequential(
        'The quick brown fox jumps over the lazy dog. Testing one two three.', opt)
    status = 'PASS' if loss_val > 0 else 'FAIL'
    print(f'[{status}] E10 (SparseHebbian deferred): loss={loss_val:.3f}')
    if status == 'FAIL': OK = False
    del m3
except Exception as e:
    print(f'[FAIL] E10: {e}'); import traceback; traceback.print_exc(); OK = False

# Test 4: Generation with rep_penalty + noise
try:
    cfg4 = V3Config(label='smoke_gen', pde_type='spectral',
                    gen_rep_penalty=1.3, gen_noise_std=0.03)
    m4 = CellAIv3(ModelParams(state_size=256, num_partitions=4, device=D), cfg4)
    opt4 = torch.optim.AdamW(m4.parameters(), lr=3e-4)
    # Quick training so the model can generate something
    for _ in range(50):
        m4.train_step_sequential('The transformer architecture learns representations', opt4)
    gen = m4.generate('The transformer architecture', max_tokens=20, temperature=0.8)
    print(f'[PASS] Generation (rep+noise): "{gen[:80]}"')
    del m4
except Exception as e:
    print(f'[FAIL] Generation: {e}'); import traceback; traceback.print_exc(); OK = False

# Test 5: AnnealedRouter adaptive lambda
try:
    from cellai_core.routing import AnnealedRouter
    router = AnnealedRouter(256, n_modalities=3, T_start=2.0, T_end=0.5,
                            anneal_steps=1000, lambda_start=0.1, lambda_end=0.001)
    router = router.to(D)
    state = torch.randn(256, device=D)
    w, logits, bal = router(state, training=True)
    # Step 0
    lam_0 = router.lambda_bal
    for _ in range(500): router.step_temperature()
    lam_500 = router.lambda_bal
    for _ in range(500): router.step_temperature()
    lam_1000 = router.lambda_bal
    status = 'PASS' if lam_0 > lam_500 > lam_1000 else 'FAIL'
    print(f'[{status}] AnnealedRouter: lam@0={lam_0:.4f} lam@500={lam_500:.4f} lam@1000={lam_1000:.4f}')
    if status == 'FAIL': OK = False
except Exception as e:
    print(f'[FAIL] AnnealedRouter: {e}'); OK = False

print()
print('ALL SMOKE TESTS PASSED' if OK else 'SOME TESTS FAILED')
