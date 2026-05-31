"""
Cell AI v3
==========
Configurable architecture for guided architecture search.

v3 is a drop-in replacement for CellAI / CellAIv2 that allows any combination of:
  - pde_type:     "dense" (original) | "spectral" (FFT-based, O(D log D))
  - hebbian_type: "full" (original)  | "sparse"   (top-k, O(D log D))
  - partition_type: "single" (original) | "multiscale" (fast+slow timescales)
  - resonance_type: "none" | "scalar" (v2 scalar phase) | "per_freq" (complex filter H)
  - use_lattice:  bool (v2 crystal lattice)
  - use_kuramoto: bool (v2 Kuramoto oscillators)

All combinations support:
  - Autoregressive generate() loop (new)
  - Multi-domain balanced training from the start (new)
  - Gradient analysis utilities (improved)
  - Complexity-annotated profiling

Architecture family
-------------------
                     text input
                         |
              UniversalEncoder (BPE cl100k_base)
                         |  (D,) embedding
                         |
             ┌──────────────────────────┐
             │  Partition System         │
             │  [dense|spectral] PDE    │
             │  [single|multiscale]     │
             └──────────┬───────────────┘
                        |  (D,) aggregate
                        |
             ┌──────────────────────────┐
             │  MemoryFormation          │
             │  [full|sparse] Hebbian    │
             │  output_proj              │
             └──────────┬───────────────┘
                        |  (D,) state
                        |
             ┌──────────────────────────┐
             │  v2 Extensions (optional) │
             │  [per_freq] Resonance    │
             │  [lattice] Crystal       │
             │  [kuramoto] Oscillator   │
             └──────────┬───────────────┘
                        |  (D,) out
                        |
                E^T projection → logits → next-token prediction

FLOP complexity (per token, D=state_size, N=num_partitions):
  Dense PDE:    O(N × D²)
  Spectral PDE: O(N × D log D)
  Full Hebbian: O(D²)
  Sparse Hebbian: O(D × k)  where k = D/8
  MemoryFormation: O(T × D)  where T=window (non-differentiable, fast)
  Resonance (scalar): O(D log D)
  Resonance (per_freq): O(D log D)
  Lattice: O(K³ × D)  K=3 → 27D
  Kuramoto: O(N² + N×D)
  Vocab projection: O(V × D)  V=100,277 (bottleneck, same for all variants)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base        import ModelParams, CellularPDE
from cellai_core.encoder     import UniversalEncoder
from cellai_core.memory      import MemoryFormation
from cellai_core.partition   import PartitionManager
from cellai_core.utils       import set_seed
from cellai_core.memory      import MetaplasticityLayer     # original (full)
from cellai_core.sparse_hebbian import SparseHebbian
from cellai_core.spectral_pde   import SpectralPDE
from cellai_core.multiscale     import MultiScalePartitionManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frequency FFT resonance (v2 upgrade)
# ─────────────────────────────────────────────────────────────────────────────

class PerFreqResonance(nn.Module):
    """
    Learnable complex spectral filter H ∈ C^(D//2+1) with MULTIPLICATIVE GATING.

    Previous design (additive residual) suffered a dead-initialisation trap:
      state_out = state + scale * resonance(state)
    When scale≈0, the module output is near zero and receives near-zero gradient.

    Fix: multiplicative SiGLU-style gate:
      gate = sigmoid(linear_gate(state))          ← initialised to ~0.5, always non-zero grad
      resonance_out = irfft(fft(state) * H)       ← spectral filter
      state_out = resonance_out * gate            ← element-wise gating

    This ensures non-zero gradient to resonance parameters from the first step.
    The gate is initialised so that sigmoid(0) = 0.5, providing 50% pass-through
    and 25% maximum gradient on the sigmoid (vs 0 for the previous additive residual).

    Complexity: O(D log D) for FFT + O(D) for gate linear
    """

    def __init__(self, state_size: int):
        super().__init__()
        F_sz = state_size // 2 + 1
        self.D     = state_size
        # Spectral filter parameters (initialised to near-identity)
        self.log_mag = nn.Parameter(torch.zeros(F_sz))
        self.phase   = nn.Parameter(torch.randn(F_sz) * 0.1)  # small random phase: breaks symmetry
        # Multiplicative gate: maps state → gate values ∈ (0,1)
        # bias = +3.0 → sigmoid(3) = 0.95: high pass-through at init
        # This prevents the gate from saturating to 0 in early training
        self.gate_proj = nn.Linear(state_size, state_size, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 3.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        s     = state.float().reshape(-1)                  # (D,)
        F_s   = tfft.rfft(s, n=self.D)                    # (F,) complex
        mag   = torch.sigmoid(self.log_mag) * 2.0          # ∈ (0, 2)
        H     = torch.polar(mag, self.phase)               # (F,) complex
        s_flt = torch.fft.irfft(F_s * H, n=self.D)        # (D,) spectral-filtered
        gate  = torch.sigmoid(self.gate_proj(state))       # (D,) ∈ (0,1) — always ~0.5 at init
        return s_flt * gate                                # (D,) gated output


# ─────────────────────────────────────────────────────────────────────────────
# v2 Crystal Lattice and Kuramoto (imported-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class CrystalLattice(nn.Module):
    """Discrete K×K×K lattice interaction (same as v2)."""
    def __init__(self, state_size: int, K: int = 3):
        super().__init__()
        self.K = K
        self.T   = nn.Parameter(torch.randn(K, K, K) * 0.05)
        self.Phi = nn.Parameter(torch.randn(K, K, K, state_size) * 0.05)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        s    = state.float().reshape(-1)
        proj = torch.einsum("ijk d, d -> ijk", self.Phi, s)
        return torch.einsum("ijk, ijk d -> d", self.T * proj, self.Phi)


# ─────────────────────────────────────────────────────────────────────────────
# v3 Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V3Config:
    """All architecture switches for CellAIv3."""
    pde_type:       str  = "dense"       # "dense" | "spectral"
    hebbian_type:   str  = "full"        # "full"  | "sparse"
    partition_type: str  = "single"      # "single" | "multiscale"
    resonance_type: str  = "none"        # "none" | "scalar" | "per_freq"
    use_lattice:    bool = False
    use_kuramoto:   bool = False
    # Multiscale parameters
    N_fast:  int   = 4
    D_fast:  int   = 128
    N_slow:  int   = 2
    D_slow:  int   = 256
    K_slow:  int   = 8
    # Hebbian sparse parameters
    k_frac:  float = 0.125
    # Generation parameters
    gen_rep_penalty:  float = 1.3    # repetition penalty (1.0 = off)
    gen_noise_std:    float = 0.03   # stochastic state noise std (0.0 = off)
    # Extra description for the paper
    label:   str   = "v3_base"


# ─────────────────────────────────────────────────────────────────────────────
# CellAIv3
# ─────────────────────────────────────────────────────────────────────────────

class CellAIv3(nn.Module):
    """
    Configurable CellularAI model for architecture search.

    Combines any of:
      - SpectralPDE or DensePDE
      - SparseHebbian or FullHebbian (MetaplasticityLayer)
      - SingleScale or MultiScale partitions
      - PerFreqResonance or ScalarResonance or none
      - Crystal Lattice (optional)
      - Kuramoto oscillators (optional)
    """

    VERSION = "v3"

    def __init__(
        self,
        params: Optional[ModelParams] = None,
        cfg:    Optional[V3Config]    = None,
    ):
        super().__init__()
        self.params = params or ModelParams()
        self.cfg    = cfg    or V3Config()
        set_seed(self.params.seed)
        self.device = torch.device(self.params.device)

        D = self.params.state_size
        N = self.params.num_partitions

        # ── Encoder (shared across all versions) ──────────────────────────
        self.encoder = UniversalEncoder(
            state_size = D,
            vocab      = self.params.encoder_vocab,
        )

        # ── Partition system ──────────────────────────────────────────────
        if self.cfg.partition_type == "multiscale":
            self.partitions = MultiScalePartitionManager(
                D_out    = D,
                N_fast   = self.cfg.N_fast,
                D_fast   = self.cfg.D_fast,
                N_slow   = self.cfg.N_slow,
                D_slow   = self.cfg.D_slow,
                K_slow   = self.cfg.K_slow,
                device   = str(self.device),
                pde_type = self.cfg.pde_type,   # pass spectral/dense through
            )
            self._multiscale = True
        else:
            if self.cfg.pde_type == "spectral":
                self.partitions = _SpectralPartitionManager(self.params)
            else:
                self.partitions = PartitionManager(self.params)
            self._multiscale = False

        # ── Memory formation ──────────────────────────────────────────────
        self.memory_formation = MemoryFormation(
            memory_size = D,
            time_window = self.params.tau_memory,
            alpha       = self.params.alpha,
            beta        = self.params.beta,
            omega       = self.params.omega,
        )

        # ── Hebbian plasticity ────────────────────────────────────────────
        if self.cfg.hebbian_type == "sparse":
            self.metaplasticity = SparseHebbian(D, k_frac=self.cfg.k_frac)
        else:
            self.metaplasticity = MetaplasticityLayer(D)

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Linear(D, D)

        # ── v2/v3 extensions ──────────────────────────────────────────────
        if self.cfg.resonance_type == "per_freq":
            self.resonance = PerFreqResonance(D)
        elif self.cfg.resonance_type == "scalar":
            self.resonance = _ScalarResonance(D)
        else:
            self.resonance = None

        if self.cfg.use_lattice:
            self.lattice = CrystalLattice(D)
            self.log_alpha_lat = nn.Parameter(torch.tensor(-3.0))
        else:
            self.lattice = None

        if self.cfg.use_kuramoto:
            self.natural_freq = nn.Parameter(torch.randn(N) * 0.1)
            self.register_buffer("phase", torch.zeros(N))
            self.osc_proj     = nn.Linear(N, D, bias=False)
            self.log_alpha_osc = nn.Parameter(torch.tensor(-3.0))
        else:
            self.osc_proj = None

        # log_alpha_ext removed: it caused the optimizer to suppress all extensions
        # to near-zero during early training (exp(-2) = 0.135 → drifts → exp(-10) ≈ 0).
        # Extensions now apply unconditionally (lattice/kuramoto keep their own alphas).

        self.to(self.device)

    # ------------------------------------------------------------------

    def _kuramoto_step(self) -> torch.Tensor:
        N = self.params.num_partitions
        phase_diff = self.phase.unsqueeze(0) - self.phase.unsqueeze(1)
        dtheta = 0.1 * torch.sin(phase_diff).sum(dim=1) + self.natural_freq
        self.phase = (self.phase + self.params.dt * dtheta).detach()
        return self.osc_proj(torch.cos(self.phase))

    def encode_input(self, text: str) -> torch.Tensor:
        return self.encoder.encode_pooled(text, device=self.device)

    def cellular_step(self, encoded: torch.Tensor) -> torch.Tensor:
        """One full forward pass through all components."""
        # Partition step
        self.partitions.step(encoded)
        agg = self.partitions.aggregate()

        # Memory + Hebbian
        memory = self.memory_formation(encoded, agg)
        out    = self.metaplasticity(agg, memory, encoded)
        base   = self.output_proj(out)

        # Extensions — applied unconditionally (no log_alpha_ext suppressor)
        if self.resonance is not None:
            # PerFreqResonance returns s_flt * gate (gate≈0.95 at init).
            # Additive residual so base is always reachable by gradient.
            base = base + self.resonance(base)
        if self.lattice is not None:
            base = base + torch.exp(self.log_alpha_lat) * self.lattice(base)
        if self.osc_proj is not None:
            base = base + torch.exp(self.log_alpha_osc) * self._kuramoto_step()

        return base

    def forward(self, text: str) -> torch.Tensor:
        encoded = self.encode_input(text)
        return self.cellular_step(encoded)

    def chat(self, prompt: str) -> str:
        with torch.no_grad():
            state = self.forward(prompt)
        return self.encoder.decode_logits(state, top_k=40)

    def generate(
        self,
        prompt:          str,
        max_tokens:      int   = 64,
        temperature:     float = 0.8,
        top_p:           float = 0.9,
        rep_penalty:     Optional[float] = None,   # repetition penalty (None = use cfg default)
        noise_std:       Optional[float] = None,   # state noise std (None = use cfg default)
        reset_state:     bool  = True,
    ) -> str:
        """
        Autoregressive generation with repetition penalty and stochastic state injection.

        Fixes for degenerate attractor:
          1. Repetition penalty (Keskar et al. 2019 CTRL): divide logits of recently
             generated tokens by rep_penalty (>1.0 suppresses repetition).
          2. Stochastic state injection: add small Gaussian noise to the cellular state
             after each step, providing an escape from fixed-point attractors.

        Args:
            prompt:      prompt text to condition the cellular state
            max_tokens:  number of tokens to generate
            temperature: sampling temperature (lower = greedier)
            top_p:       nucleus sampling cumulative probability cutoff
            rep_penalty: repetition penalty coefficient (default: cfg.gen_rep_penalty)
            noise_std:   Gaussian noise std on cellular state (default: cfg.gen_noise_std)
            reset_state: if False, keep partitions/memory (use after external warm-up)

        Returns:
            Generated text string (prompt + continuation)
        """
        self.eval()

        _rep  = rep_penalty if rep_penalty is not None else self.cfg.gen_rep_penalty
        _noise = noise_std  if noise_std  is not None else self.cfg.gen_noise_std

        tokens = self.encoder.tokenize(prompt)
        if not tokens:
            tokens = [0]

        if reset_state:
            self.partitions.reset()
            self.memory_formation.reset()

        tok_ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
        embs = self.encoder.embedding(tok_ids) * self.encoder._scale  # (L, D)

        # Warm up on prompt (full cellular_step = same path as training / NTP)
        with torch.no_grad():
            for i in range(len(tokens)):
                _ = self.cellular_step(embs[i])

        # Autoregressive generation
        generated_ids: List[int] = []
        # track recently generated tokens for repetition penalty (last 32)
        recent_window = 32
        current_emb = embs[-1] if len(embs) > 0 else self.encoder.embedding.weight[0] * self.encoder._scale

        W_snap = self.metaplasticity.W.data.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
                state = self.cellular_step(current_emb)

                # Stochastic state injection — breaks fixed-point attractors
                if _noise > 0:
                    state = state + torch.randn_like(state) * _noise

                # Logits over vocabulary
                logits = state @ self.encoder.embedding.weight.t()   # (V,)

                # Repetition penalty: divide logits of recently seen tokens
                if _rep > 1.0 and generated_ids:
                    recent = generated_ids[-recent_window:]
                    for tok in set(recent):
                        if logits[tok] > 0:
                            logits[tok] /= _rep
                        else:
                            logits[tok] *= _rep

                # Temperature scaling
                logits = logits / max(temperature, 1e-5)

                # Nucleus (top-p) sampling
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cum_probs - sorted_probs) < top_p
                sorted_probs[~cutoff] = 0.0
                sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-9)

                # Sample
                next_idx = torch.multinomial(sorted_probs, 1)
                next_tok = sorted_idx[next_idx].item()
                generated_ids.append(next_tok)

                # Next input embedding
                current_emb = self.encoder.embedding.weight[next_tok] * self.encoder._scale

        # Restore Hebbian weights to pre-generation state
        self.metaplasticity.W.data.copy_(W_snap)

        # Decode via encoder's detokenize (handles edge cases)
        gen_text = self.encoder.detokenize(generated_ids)
        return prompt + gen_text

    def train_step_sequential(
        self,
        text:        str,
        optimizer:   torch.optim.Optimizer,
        segment_len: int  = 64,
        reset_state: bool = True,
    ) -> float:
        """Next-token prediction with truncated BPTT. Compatible with v1 API."""
        was_training = self.training
        self.train()

        if reset_state:
            self.partitions.reset()
            self.memory_formation.reset()

        tokens = self.encoder.tokenize(text)
        if len(tokens) < 2:
            if not was_training: self.eval()
            return 0.0

        optimizer.zero_grad()
        total_loss, token_count = 0.0, 0

        tok_ids = torch.tensor(tokens, dtype=torch.long, device=self.device)

        for seg_start in range(0, len(tokens) - 1, segment_len):
            seg_end  = min(seg_start + segment_len, len(tokens) - 1)
            seg_ids  = tok_ids[seg_start : seg_end + 1]
            seg_embs = self.encoder.embedding(seg_ids) * self.encoder._scale

            seg_loss = torch.tensor(0.0, device=self.device)

            for t_local in range(seg_end - seg_start):
                inp = seg_embs[t_local]

                # Use cellular_step so ALL components (resonance, lattice, etc.)
                # participate in the gradient computation.  Previously the loop
                # bypassed extensions by calling partitions/memory/metaplasticity
                # individually, so resonance was never trained.
                state = self.cellular_step(inp)

                logits = state @ self.encoder.embedding.weight.t()
                target = seg_ids[t_local + 1].unsqueeze(0)
                seg_loss = seg_loss + F.cross_entropy(logits.unsqueeze(0), target)
                token_count += 1

            (seg_loss / max(seg_end - seg_start, 1)).backward()
            total_loss += seg_loss.item()

            # Apply deferred Hebbian update AFTER backward (prevents autograd graph corruption)
            if hasattr(self.metaplasticity, 'apply_pending_update'):
                self.metaplasticity.apply_pending_update()

            # Detach state for next segment (truncated BPTT)
            if self._multiscale:
                self.partitions.detach_state()
            else:
                self.partitions._buffers["state"] = (
                    self.partitions._buffers["state"].detach())  # type: ignore

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        if not was_training:
            self.eval()
        return total_loss / max(token_count, 1)

    def benchmark(self, n_samples: int = 200, text: str = "benchmark sample") -> Dict[str, float]:
        with torch.no_grad():
            for _ in range(10):
                self.forward(text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_samples):
                self.forward(text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "label":         self.cfg.label,
            "ms_per_sample": elapsed * 1000 / n_samples,
            "samples_per_s": n_samples / elapsed,
        }

    def get_info(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "label":          self.cfg.label,
            "pde_type":       self.cfg.pde_type,
            "hebbian_type":   self.cfg.hebbian_type,
            "partition_type": self.cfg.partition_type,
            "resonance_type": self.cfg.resonance_type,
            "n_params":       n_params,
            "state_size":     self.params.state_size,
            "num_partitions": self.params.num_partitions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: spectral partition wrapper and scalar resonance
# ─────────────────────────────────────────────────────────────────────────────

class _SpectralPartitionManager(nn.Module):
    """PartitionManager-compatible wrapper using SpectralPDE instead of CellularPDE."""

    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.N = params.num_partitions
        self.D = params.state_size
        dev = torch.device(params.device)
        self.pde = SpectralPDE(self.N, self.D).to(dev)
        self.register_buffer("state", torch.zeros(self.N, self.D, device=dev))

    def step(self, inp: torch.Tensor) -> torch.Tensor:
        new_state = self.pde.step(self._buffers["state"], inp, self.params)
        self._buffers["state"] = new_state
        return new_state

    def aggregate(self) -> torch.Tensor:
        return self._buffers["state"].mean(dim=0)

    def reset(self) -> None:
        # Always allocate fresh leaf tensor — prevents grad_fn contamination across passes
        self._buffers["state"] = torch.zeros(
            self.N, self.D,
            device=self._buffers["state"].device,
            dtype=torch.float32,
        )


class _ScalarResonance(nn.Module):
    """v2-compatible scalar phase resonance."""
    def __init__(self, state_size: int):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(1))
        self.D = state_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        s = state.float()
        F_s   = tfft.fft(s)
        F_mod = F_s * torch.exp(1j * self.phase)
        return torch.real(tfft.ifft(F_mod))
