"""
cellai_core.memory
==================
Memory formation and metaplasticity for Cell AI.

Memory kernel:
    K(t) = α · exp(-β·t) · cos(ω·t)      (oscillatory exponential decay)
    w(t) = exp(-t/τ₁) - exp(-t/τ₂)       (difference-of-exponentials)

Memory formation (discrete approximation of the integral):
    M(t) ≈ Σₛ w(t-s) · I(s) + Σₛ K(t-s) · S(s)

Metaplasticity (sliding-threshold Hebbian rule):
    θ(t) ← θ(t-1) + α(M(t) - θ(t-1)) + β·M̄(t)

All operations are vectorised; no Python loops over time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------

def memory_kernel(
    time_window: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    omega: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """K(t) = α · exp(-β·t) · cos(ω·t)  shape: (time_window,)"""
    t = torch.arange(time_window, dtype=torch.float32, device=device)
    return alpha * torch.exp(-beta * t) * torch.cos(omega * t)


def weight_kernel(
    time_window: int,
    tau1: float = 20.0,
    tau2: float = 5.0,
    device: str = "cpu",
) -> torch.Tensor:
    """w(t) = exp(-t/τ₁) - exp(-t/τ₂)  shape: (time_window,)"""
    t = torch.arange(time_window, dtype=torch.float32, device=device)
    return torch.exp(-t / tau1) - torch.exp(-t / tau2)


# ---------------------------------------------------------------------------
# MemoryFormation
# ---------------------------------------------------------------------------

class MemoryFormation(nn.Module):
    """
    Circular-buffer memory integration:
        M(t) ≈ Σₛ w(t-s)·I(s) + Σₛ K(t-s)·S(s)

    Maintains a ring buffer of the last `time_window` input and state vectors.
    The weighted sum is computed with a single matmul (kernel × buffer).

    Args:
        memory_size:   D — dimensionality of the state / input
        time_window:   T — history length
        alpha, beta, omega: memory kernel parameters
    """

    def __init__(
        self,
        memory_size: int,
        time_window: int = 100,
        alpha: float = 1.0,
        beta: float = 0.5,
        omega: float = 1.0,
    ):
        super().__init__()
        self.D = memory_size
        self.T = time_window

        # Pre-computed kernels — fixed (not learned)
        w = weight_kernel(time_window)
        K = memory_kernel(time_window, alpha, beta, omega)
        # Normalise to prevent exploding memory
        w = w / (w.abs().sum() + 1e-8)
        K = K / (K.abs().sum() + 1e-8)
        self.register_buffer("w_kernel", w)      # (T,)
        self.register_buffer("K_kernel", K)      # (T,)

        # Ring buffers  (T, D)
        self.register_buffer("inp_buf", torch.zeros(time_window, memory_size))
        self.register_buffer("sta_buf", torch.zeros(time_window, memory_size))
        self._ptr: int = 0                        # write pointer

    def forward(self, inp: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Update the buffers and return the new memory vector M(t) of shape (D,).
        """
        inp_1d   = inp.reshape(self.D).detach()
        state_1d = state.reshape(self.D).detach()

        # Write into ring buffer
        self.inp_buf[self._ptr]  = inp_1d
        self.sta_buf[self._ptr]  = state_1d
        self._ptr = (self._ptr + 1) % self.T

        # Build chronological view (oldest first) by rolling
        # idx maps buffer slot → time lag (0 = most recent)
        idx = (torch.arange(self.T, device=self.w_kernel.device) + self._ptr) % self.T

        # M = w.T @ inp_buf_ordered + K.T @ sta_buf_ordered
        # inp_buf_ordered[t] is the input from t steps ago → w[t] weight
        inp_ordered = self.inp_buf[idx]            # (T, D)
        sta_ordered = self.sta_buf[idx]            # (T, D)

        # (T,) @ (T, D)  →  (D,)
        M = self.w_kernel @ inp_ordered + self.K_kernel @ sta_ordered
        return M

    def reset(self) -> None:
        """Clear buffers (call between independent sequences)."""
        self.inp_buf.zero_()
        self.sta_buf.zero_()
        self._ptr = 0


# ---------------------------------------------------------------------------
# MetaplasticityLayer
# ---------------------------------------------------------------------------

class MetaplasticityLayer(nn.Module):
    """
    Sliding-threshold plasticity gate.

    Updates a running plasticity threshold θ based on recent memory M,
    then gates the output via a Hebbian-style weight matrix:

        η(t)    = exp(-|Sᵢ - Sⱼ|)     (activity-dependent learning rate)
        H(I, θ) = σ(I - θ)             (BCM-like threshold gate)
        W(t)   += η · H · H.T          (outer product weight update)
        θ(t)    = θ + α(M - θ) + β·M̄  (sliding threshold)
        output  = W(t) · I

    Args:
        state_size:  D
        alpha:       threshold adaptation rate  (default 0.1)
        beta:        slow average contribution  (default 0.01)
    """

    def __init__(self, state_size: int, alpha: float = 0.1, beta: float = 0.01):
        super().__init__()
        self.D = state_size
        self.alpha = alpha
        self.beta = beta
        self.W = nn.Parameter(torch.randn(state_size, state_size) * 0.01)

        # Differentiable state gate: output *= sigmoid(state_proj(Si))
        # This is the gradient highway from the loss back to the CellularPDE
        # weights (pde.W and pde.E).  The gate input is Si = partitions.aggregate(),
        # which is in the autograd graph through pde.step().
        # Use small random init so dgate/dSi = σ'(*) * W_gate ≠ 0 from step 1.
        self.state_gate = nn.Linear(state_size, state_size, bias=True)
        nn.init.normal_(self.state_gate.weight, std=0.01)
        nn.init.zeros_(self.state_gate.bias)

        self.register_buffer("theta", torch.zeros(state_size))
        self.register_buffer("M_avg", torch.zeros(state_size))

    def forward(
        self,
        Si: torch.Tensor,   # (D,) — current partition state (from partitions.aggregate)
        M: torch.Tensor,    # (D,) — memory vector
        I: torch.Tensor,    # (D,) — input signal
    ) -> torch.Tensor:      # (D,)
        Si = Si.reshape(self.D)
        M  = M.reshape(self.D)
        I  = I.reshape(self.D)

        # Hebbian weight update always runs (in-place via .data — no autograd).
        # The W matrix is also a learnable nn.Parameter (receives grad via state_gate),
        # but the online Hebbian rule additionally adapts W within each sequence.
        # This is a deliberate design: the model relies on W dynamics, not just W values.
        eta = torch.exp(-torch.abs(Si.detach() - M.detach()))
        H   = torch.sigmoid(I.detach() - self.theta)
        self.W.data.add_(0.001 * torch.outer(eta * H, eta * H))
        torch.clamp_(self.W.data, -1.0, 1.0)

        # Threshold update only during training (theta is part of the learning rule,
        # not needed for stable inference once training is done).
        if self.training:
            self.M_avg = 0.99 * self.M_avg.detach() + 0.01 * M.detach()
            self.theta = (
                self.theta.detach()
                + self.alpha * (M.detach() - self.theta.detach())
                + self.beta  * self.M_avg.detach()
            )

        # Differentiable forward: Hebbian output gated by cellular state.
        # The state_gate path is the gradient channel for PDE.W / PDE.E.
        # At init, state_gate.weight=0 so gate = sigmoid(0) = 0.5.
        gate       = torch.sigmoid(self.state_gate(Si))      # (D,) ← autograd through Si
        hebbian_out = F.linear(I, self.W)                     # (D,) ← W is also learned
        return hebbian_out * gate
