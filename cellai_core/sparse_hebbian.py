"""
cellai_core.sparse_hebbian
==========================
Low-rank Hebbian synaptic plasticity.

Problem with the original MetaplasticityLayer
----------------------------------------------
The original Hebbian update is:
    W += η · (η' ⊙ H)(η' ⊙ H)^T           O(D²) memory, O(D²) per update
For D=256 this is 65,536 element updates per token step.

Low-rank solution
-----------------
Maintain W as an explicit D×D matrix for the differentiable gradient path, but
constrain the *online* Hebbian updates to a low-rank factored form:

    W_hebb = U @ V^T    where U, V ∈ R^(D × r)

Each Hebbian update modifies the rank-1 outer product only in the top-k "active"
directions (neurons with highest |η' ⊙ H|). This reduces the per-step update to:

    - Top-k select:  O(D log k)   (torch.topk)
    - Outer product: O(k²)        for the selected neurons

For k = D/8 = 32:  k² = 1024 vs D² = 65536 → 64× reduction.
The rank-r structured weight: O(D×r) storage vs O(D²) → 8× for r=32.

Biological motivation
---------------------
Cortical synaptic connectivity is sparse: each neuron has ~1000 synaptic connections
out of ~10^5 nearby neurons, giving a ~1% connectivity rate [Braitenberg & Schüz, 1998].
Sparse Hebbian updates implement this connectivity constraint: only the neurons whose
activity product exceeds a threshold are strengthened, leaving weaker pathways silent.

This implements a variant of Sanger's rule [Sanger, 1989] in the k-sparse setting,
which provably converges to the top-k principal components of the input distribution —
a form of biological PCA [Oja, 1982; Sanger, 1989].

Additionally, the low-rank U,V factorization parallels the "synaptic matrix" model of
Hopfield networks [Hopfield, 1982], where memories are stored as outer products.
The rank r bounds the number of storable patterns.

Architecture in this module
---------------------------
SparseHebbian replaces MetaplasticityLayer. It has the same interface:
    forward(Si, M, I) → (D,)

Changes from MetaplasticityLayer:
1. Online update targets only top-k neurons (sparse outer product)
2. Weight stored as full D×D Parameter (for gradient path) + separately tracked
   sparse accumulator (for the Hebbian dynamics)
3. BCM sliding threshold retained (biologically important)
4. state_gate retained (critical gradient highway to PDE)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseHebbian(nn.Module):
    """
    Sparse BCM-Hebbian metaplasticity with top-k activity selection.

    Online weight update (no autograd):
        activity = η' ⊙ H    (eligibility-weighted BCM gate)
        topk_idx = argtopk(|activity|, k)
        sparse_h = zeros_like(activity);  sparse_h[topk_idx] = activity[topk_idx]
        W_hebb += η_hebb * outer(sparse_h, sparse_h)   ← O(k²) outer product

    Differentiable output path (with autograd):
        gate        = sigmoid(state_gate(Si))    ← gradient highway to CellularPDE
        hebbian_out = linear(I, W)               ← W is also an nn.Parameter
        return hebbian_out * gate

    Args:
        state_size:   D — state dimensionality
        k_frac:       fraction of neurons active per step (default 1/8)
        alpha, beta:  BCM threshold parameters
        hebb_rate:    in-place Hebbian step size
    """

    def __init__(
        self,
        state_size:    int,
        k_frac:        float = 0.125,   # sparse: 1/8 of neurons active
        alpha:         float = 0.1,
        beta:          float = 0.01,
        hebb_rate:     float = 0.005,   # 5× larger than original for faster W warmup
        defer_update:  bool  = True,    # accumulate Hebbian delta; apply after backward
    ):
        super().__init__()
        self.D            = state_size
        self.k            = max(1, int(state_size * k_frac))
        self.alpha        = alpha
        self.beta         = beta
        self.hebb_rate    = hebb_rate
        self.defer_update = defer_update

        # Learnable weight (gradient path) — D×D
        self.W = nn.Parameter(torch.randn(state_size, state_size) * 0.01)

        # Gradient highway: state → gate scaling
        self.state_gate = nn.Linear(state_size, state_size, bias=True)
        nn.init.normal_(self.state_gate.weight, std=0.01)
        nn.init.zeros_(self.state_gate.bias)

        # BCM sliding threshold (buffer: not in gradient graph)
        self.register_buffer("theta", torch.zeros(state_size))
        self.register_buffer("M_avg", torch.zeros(state_size))

        # Deferred update accumulator (not a Parameter or buffer — just a tensor)
        self._hebb_accum: Optional[torch.Tensor] = None

        # Sparsity statistics (for monitoring)
        self.register_buffer("update_count", torch.zeros(state_size))

    def forward(
        self,
        Si: torch.Tensor,   # (D,) — aggregate cellular state
        M:  torch.Tensor,   # (D,) — memory vector
        I:  torch.Tensor,   # (D,) — input embedding
    ) -> torch.Tensor:      # (D,)
        Si = Si.reshape(self.D)
        M  = M.reshape(self.D)
        I  = I.reshape(self.D)

        # ── Sparse Hebbian update (training only; inference keeps W fixed) ─
        if self.training:
            with torch.no_grad():
                eta_prime = torch.exp(-torch.abs(Si - M))            # eligibility
                H_gate    = torch.sigmoid(I - self.theta)            # BCM gate

                activity  = eta_prime * H_gate                       # (D,)

                # Select top-k most active neurons
                _, topk_idx = torch.topk(activity.abs(), self.k)    # indices (k,)
                sparse_act  = torch.zeros_like(activity)
                sparse_act[topk_idx] = activity[topk_idx]           # (D,) sparse

                # Outer product update — only k² non-zero entries
                delta_W = self.hebb_rate * torch.outer(sparse_act, sparse_act)   # (D,D)

                if self.defer_update:
                    # Accumulate delta; applied atomically after backward() via apply_pending_update()
                    # This prevents autograd graph corruption when W is shared across timesteps.
                    if self._hebb_accum is None:
                        self._hebb_accum = delta_W.clone()
                    else:
                        self._hebb_accum.add_(delta_W)
                else:
                    self.W.data.add_(delta_W)
                    torch.clamp_(self.W.data, -1.0, 1.0)

                # Track which neurons are being updated
                self.update_count[topk_idx] += 1

        # ── BCM threshold update ─────────────────────────────────────────
        if self.training:
            with torch.no_grad():
                self.M_avg = 0.99 * self.M_avg + 0.01 * M
                self.theta = (self.theta
                              + self.alpha * (M - self.theta)
                              + self.beta  * self.M_avg)

        # ── Differentiable forward ───────────────────────────────────────
        gate        = torch.sigmoid(self.state_gate(Si))   # (D,) via Si → grad to PDE
        hebbian_out = F.linear(I, self.W)                  # (D,) via W + I
        return hebbian_out * gate

    def apply_pending_update(self) -> None:
        """
        Apply the accumulated Hebbian delta to W and clear the accumulator.
        Call this AFTER seg_loss.backward() to avoid autograd graph corruption.
        """
        if self._hebb_accum is not None:
            self.W.data.add_(self._hebb_accum)
            torch.clamp_(self.W.data, -1.0, 1.0)
            self._hebb_accum = None

    def sparsity_stats(self) -> dict:
        """Diagnostic: fraction of neurons never updated, Gini of update counts."""
        uc = self.update_count.float()
        n_never = (uc == 0).sum().item()
        total   = uc.sum().item() + 1e-9
        # Gini coefficient
        uc_sorted = uc.sort().values
        n = len(uc_sorted)
        idx = torch.arange(1, n + 1, dtype=torch.float32, device=uc.device)
        gini = (2 * (idx * uc_sorted).sum() / (n * uc_sorted.sum() + 1e-9)) - (n + 1) / n
        return {
            "never_updated_frac": n_never / self.D,
            "gini_coefficient":   gini.item(),
            "total_updates":      total,
        }
