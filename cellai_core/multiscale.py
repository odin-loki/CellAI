"""
cellai_core.multiscale
======================
Multi-timescale partition system.

Motivation
----------
Biological neural circuits operate across multiple timescales simultaneously:
  - Fast rhythms (gamma, 30-100 Hz): fine-grained local feature processing
  - Slow rhythms (theta, 4-8 Hz): long-range integration and memory consolidation
  [Buzsáki & Draguhn, 2004; Mayer & Sun, 2011]

In the cellular language model context:
  - Fast partitions (update every token): character/subword-level features
  - Slow partitions (update every K tokens): sentence/paragraph-level context

The hierarchical organisation has O(N_fast × D_f² + N_slow × D_s²) cost per token
when using dense PDE, or O(N_fast × D_f log D_f + N_slow × D_s log D_s) with SpectralPDE.

Design
------
MultiScalePartitionManager maintains two separate PartitionManagers:
  - fast_pm:  N_fast partitions, state_size=D_fast, update every token
  - slow_pm:  N_slow partitions, state_size=D_slow, update every K tokens

Cross-scale communication:
  - Fast → Slow (every K tokens): slow partitions receive the mean of fast states,
    downsampled to D_slow via a learned linear projection
  - Slow → Fast (every token):  fast partitions receive a "context" from slow states,
    upsampled to D_fast via a learned linear projection

The final aggregate is the concatenation of fast-aggregate (D_fast,) and
slow-aggregate (D_slow,), projected back to D via a learned linear (needed to
match the downstream state_size).

Complexity per token:
  - Fast PDE: O(N_fast × D_fast²)  [or O(N_fast × D_fast log D_fast) with SpectralPDE]
  - Slow PDE: O(N_slow × D_slow²)  [updated K-fold less often]
  - Cross-scale projection: O(D_fast × D_slow) per token (linear layer forward)
  - Aggregate projection: O(D) per token
  Total: O(N_fast × D_fast² + D_fast × D_slow + D)  ≤ O(N_dense × D²)  when D_fast<D

Default: N_fast=4, D_fast=128, N_slow=2, D_slow=256, K=8
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base         import CellularPDE, ModelParams
from cellai_core.partition    import PartitionManager
from cellai_core.spectral_pde import SpectralPDE


class MultiScalePartitionManager(nn.Module):
    """
    Two-speed cellular partition system.

    Args:
        D_out:    output state size (must match downstream model's state_size)
        N_fast:   number of fast partitions
        D_fast:   state size of fast partitions
        N_slow:   number of slow partitions
        D_slow:   state size of slow partitions
        K_slow:   slow-partition update frequency (every K tokens)
        device:   torch device
    """

    def __init__(
        self,
        D_out:    int = 256,
        N_fast:   int = 4,
        D_fast:   int = 128,
        N_slow:   int = 2,
        D_slow:   int = 256,
        K_slow:   int = 8,
        device:   Optional[str] = None,
        pde_type: str = "dense",   # "dense" | "spectral"
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.D_out    = D_out
        self.N_fast   = N_fast
        self.D_fast   = D_fast
        self.N_slow   = N_slow
        self.D_slow   = D_slow
        self.K_slow   = K_slow
        self.pde_type = pde_type
        self.dev      = torch.device(device)

        # Fast and slow partition systems — spectral or dense PDE
        fast_params = ModelParams(num_partitions=N_fast, state_size=D_fast, device=device)
        slow_params = ModelParams(num_partitions=N_slow, state_size=D_slow, device=device)

        if pde_type == "spectral":
            self.fast_pm = _SpectralPartitionWrapper(fast_params)
            self.slow_pm = _SpectralPartitionWrapper(slow_params)
        else:
            self.fast_pm = PartitionManager(fast_params)
            self.slow_pm = PartitionManager(slow_params)

        # Cross-scale projections
        # Input: D_out → D_fast and D_out → D_slow
        self.inp_to_fast = nn.Linear(D_out,  D_fast, bias=False).to(self.dev)
        self.inp_to_slow = nn.Linear(D_out,  D_slow, bias=False).to(self.dev)

        # Slow context → fast state (additive residual)
        self.slow_to_fast = nn.Linear(D_slow, D_fast, bias=True).to(self.dev)
        nn.init.zeros_(self.slow_to_fast.weight)   # start as zero: no slow→fast initially

        # Fast state → slow update (used at slow-update steps)
        self.fast_to_slow = nn.Linear(D_fast, D_slow, bias=True).to(self.dev)
        nn.init.normal_(self.fast_to_slow.weight, std=0.01)

        # Aggregate projection: (D_fast + D_slow) → D_out
        self.aggregate_proj = nn.Linear(D_fast + D_slow, D_out, bias=True).to(self.dev)

        # Step counter for K-fold update
        self.register_buffer("_step_count", torch.tensor(0, device=self.dev))

    def step(self, inp: torch.Tensor) -> None:
        """
        One multi-scale cellular step.

        1. Project input to fast/slow dimensions
        2. Update fast partitions every token (with slow context as residual)
        3. Update slow partitions every K tokens (with fast aggregate as input)

        Args:
            inp: (D_out,) input embedding
        """
        inp = inp.to(self.dev)

        # Project input to fast and slow spaces
        inp_fast = self.inp_to_fast(inp)   # (D_fast,)
        inp_slow = self.inp_to_slow(inp)   # (D_slow,)

        # Slow context (from previous slow state) added to fast input
        slow_agg  = self.slow_pm.aggregate()          # (D_slow,)
        slow_ctx  = torch.tanh(self.slow_to_fast(slow_agg))  # (D_fast,)

        # Fast update: combined input + slow context
        self.fast_pm.step(inp_fast + slow_ctx)

        # Slow update every K tokens
        cnt = int(self._step_count.item())
        if cnt % self.K_slow == 0:
            fast_agg = self.fast_pm.aggregate()                  # (D_fast,)
            slow_inp = inp_slow + self.fast_to_slow(fast_agg)    # (D_slow,)
            self.slow_pm.step(slow_inp)

        self._step_count += 1

    def aggregate(self) -> torch.Tensor:
        """
        Aggregate both timescales → (D_out,) via learned projection.
        """
        fast_agg = self.fast_pm.aggregate()  # (D_fast,)
        slow_agg = self.slow_pm.aggregate()  # (D_slow,)
        cat = torch.cat([fast_agg, slow_agg], dim=0)  # (D_fast + D_slow,)
        return self.aggregate_proj(cat)      # (D_out,)

    def reset(self) -> None:
        self.fast_pm.reset()
        self.slow_pm.reset()
        self._step_count.zero_()

    def detach_state(self):
        """Detach both partition states for truncated BPTT."""
        self.fast_pm._buffers["state"] = self.fast_pm._buffers["state"].detach()
        self.slow_pm._buffers["state"] = self.slow_pm._buffers["state"].detach()


# ─────────────────────────────────────────────────────────────────────────────
# SpectralPDE wrapper with PartitionManager-compatible interface
# ─────────────────────────────────────────────────────────────────────────────

class _SpectralPartitionWrapper(nn.Module):
    """Drop-in PartitionManager replacement using SpectralPDE (O(D log D) per step)."""

    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.N      = params.num_partitions
        self.D      = params.state_size
        dev         = torch.device(params.device)
        self.pde    = SpectralPDE(self.N, self.D).to(dev)
        self.register_buffer("state", torch.zeros(self.N, self.D, device=dev))

    def step(self, inp: torch.Tensor) -> torch.Tensor:
        new_state = self.pde.step(self._buffers["state"], inp, self.params)
        self._buffers["state"] = new_state
        return new_state

    def aggregate(self) -> torch.Tensor:
        return self._buffers["state"].mean(dim=0)

    def reset(self) -> None:
        self._buffers["state"] = torch.zeros(
            self.N, self.D,
            device=self._buffers["state"].device,
            dtype=torch.float32,
        )
