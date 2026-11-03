"""Mechanistic checks: dense-PDE zero state, SpectralPDE sensitivity, resonance = cos(phi)*s, lattice rank, Kuramoto gradient, Hebbian eval-mode mutation and rank collapse, SparseHebbian eval behaviour, padding_idx (sections 2.4, 8)."""
import sys
import os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import torch, torch.nn.functional as F, math, json
torch.set_num_threads(1)
from cellai_core.base import ModelParams, CellularPDE
from cellai_core.spectral_pde import SpectralPDE
from cellai_core.memory import MetaplasticityLayer
from cellai_core.sparse_hebbian import SparseHebbian
from v1.cell_ai import CellAI
from v2.cell_ai_v2 import CellAIv2, ResonanceSystem, CrystalLattice
out = {}
# --- Dense PDE zero-state, via v1 model (realistic inputs = embedding*sqrt(D)) ---
torch.manual_seed(0)
m = CellAI(ModelParams(device="cpu")); m.eval()
emb = m.encoder.embedding.weight
ids1 = torch.randint(1, 100000, (256,)); ids2 = torch.randint(1, 100000, (256,))
def run_pde(ids, eta, seed=123, model=m):
    p = ModelParams(device="cpu", eta=eta)
    S = torch.zeros(4, 256)
    g = torch.manual_seed(seed)
    with torch.no_grad():
        for t in ids:
            x = emb[t] * m.encoder._scale
            S = m.partitions.pde.step(S, x, p)
    return S
S0 = run_pde(ids1, 0.0); out["dense_noise0_absmax"] = float(S0.abs().max())
Sa = run_pde(ids1, 0.01, 123); Sb = run_pde(ids2, 0.01, 123)
out["dense_samenoise_rel_diff_agg"] = float((Sa.mean(0)-Sb.mean(0)).norm()/Sa.mean(0).norm())
out["dense_samenoise_rel_diff_full"] = float((Sa-Sb).norm()/Sa.norm())
Sc = run_pde(ids1, 0.01, 999)
out["dense_diffnoise_sameinput_rel_diff_agg"] = float((Sa.mean(0)-Sc.mean(0)).norm()/Sa.mean(0).norm())
out["dense_state_norm"] = float(Sa.norm())
# --- Spectral PDE sensitivity (N=4,D=256 default gamma .05) ---
torch.manual_seed(0)
sp = SpectralPDE(4, 256); p = ModelParams(device="cpu")
def run_sp(ids, mod=sp, D=256):
    S = torch.zeros(mod.N, D)
    with torch.no_grad():
        for t in ids:
            x = emb[t] * m.encoder._scale if D==256 else torch.randn(D)
            S = mod.step(S, x, p)
    return S
A = run_sp(ids1); B = run_sp(ids2)
out["spectral_absmean_after256"] = float(A.abs().mean()); out["spectral_norm_after256"] = float(A.norm()); out["spectral_aggnorm"] = float(A.mean(0).norm())
out["spectral_rel_diff_agg"] = float((A.mean(0)-B.mean(0)).norm()/A.mean(0).norm())
out["spectral_rel_diff_full"] = float((A-B).norm()/A.norm())
# with unit-variance inputs instead of sqrt(D)-scaled embedding
def run_sp2(seed):
    g = torch.Generator().manual_seed(seed); S = torch.zeros(4,256)
    with torch.no_grad():
        for _ in range(256): S = sp.step(S, torch.randn(256, generator=g), p)
    return S
A2, B2 = run_sp2(1), run_sp2(2)
out["spectral_unitinput_rel_diff_agg"] = float((A2.mean(0)-B2.mean(0)).norm()/A2.mean(0).norm())
out["spectral_unitinput_aggnorm"] = float(A2.mean(0).norm())
# --- Resonance / lattice ---
r = ResonanceSystem(256); s = torch.randn(256)
errs = []
for phi in [0.0, 0.3, 1.0, 2.0]:
    r.phase.data.fill_(phi); errs.append(float((r(s) - math.cos(phi)*s).abs().max()))
out["resonance_cos_maxerr"] = max(errs)
L = CrystalLattice(256); M = torch.einsum("ijkd,ijk,ijke->de", L.Phi, L.T, L.Phi)
out["lattice_linear_err"] = float((L(s)-M@s).abs().max()); out["lattice_rank"] = int(torch.linalg.matrix_rank(M)); out["lattice_sym"] = float((M-M.T).abs().max())
# --- Kuramoto grad ---
m2 = CellAIv2(ModelParams(device="cpu")); m2.train()
x = m2.encoder.embedding.weight[5]*m2.encoder._scale
loss = 0
for _ in range(5): loss = loss + m2.cellular_step(x.detach()).pow(2).sum()
loss.backward()
out["kuramoto_natural_freq_grad"] = None if m2.natural_freq.grad is None else float(m2.natural_freq.grad.abs().max())
out["v2_phase_grad"] = None if m2.resonance.phase.grad is None else float(m2.resonance.phase.grad.abs().max())
# --- Metaplasticity: eval-mode mutation + rank collapse ---
ml = MetaplasticityLayer(256); ml.eval()
W0 = ml.W.data.clone()
with torch.no_grad():
    ml(torch.randn(256), torch.randn(256), torch.randn(256))
out["metaplasticity_eval_mutates"] = float((ml.W.data-W0).abs().max())
inc = ml.W.data - W0; out["metaplasticity_increment_min"] = float(inc.min())
ml2 = MetaplasticityLayer(256); ml2.train()
with torch.no_grad():
    for _ in range(2000): ml2(torch.randn(256), torch.randn(256), torch.randn(256))
sv = torch.linalg.svdvals(ml2.W.data); out["metaplasticity_top_sv_energy_2000"] = float(sv[0]**2/(sv**2).sum())
# with realistic v1 model over 2000 random tokens (chat-style eval)
m3 = CellAI(ModelParams(device="cpu")); m3.eval()
with torch.no_grad():
    for t in torch.randint(1,100000,(2000,)):
        m3.cellular_step(m3.encoder.embedding.weight[t]*m3.encoder._scale)
sv = torch.linalg.svdvals(m3.metaplasticity.W.data); out["v1_eval_W_top_sv_energy_2000tok"] = float(sv[0]**2/(sv**2).sum())
# --- SparseHebbian eval-mode ---
sh = SparseHebbian(256); sh.eval(); W0 = sh.W.data.clone()
with torch.no_grad(): sh(torch.randn(256), torch.randn(256), torch.randn(256))
out["sparsehebb_eval_change"] = float((sh.W.data-W0).abs().max()); out["sparsehebb_eval_accum"] = sh._hebb_accum is None
sh.train()
with torch.no_grad(): sh(torch.randn(256), torch.randn(256), torch.randn(256))
out["sparsehebb_train_accum_shape"] = list(sh._hebb_accum.shape); out["sparsehebb_nnz"] = int((sh._hebb_accum!=0).sum())
# --- padding_idx ---
import tiktoken
enc = tiktoken.get_encoding("cl100k_base"); out["tok0"] = enc.decode([0])
m4 = CellAI(ModelParams(device="cpu", state_size=64, num_partitions=2))
out["emb_row0_init_norm"] = float(m4.encoder.embedding.weight[0].norm())
opt = torch.optim.AdamW(m4.parameters(), lr=1e-3)
for _ in range(3):
    m4.train_step_sequential("Hello! World! Yes!! " * 5, opt)
out["emb_row0_after_train_norm"] = float(m4.encoder.embedding.weight[0].norm())
print(json.dumps(out, indent=1))
