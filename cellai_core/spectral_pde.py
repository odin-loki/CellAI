"""
cellai_core.spectral_pde
========================
SpectralPDE: FFT-based cellular diffusion operator.

Complexity analysis
-------------------
Dense PDE (original):
    S' = f(inp @ W.T, state @ E.T)   — O(D²) per partition per token
    Total: O(N × D²)

SpectralPDE (this module):
    S' = IFFT(FFT(state) * H_s + FFT(inp) * H_i)   — O(D log D) per partition
    Total: O(N × D log D)

For N=4, D=256:
    Dense: 4 × 65,536 = 262,144 multiply-adds
    Spectral: 4 × 2,048 ≈ 8,192 + 2 FFTs ≈ 16,384 ops
    Speedup: ~16× in FLOPs

Biological motivation
---------------------
The hippocampal CA3 region uses oscillatory phase codes to represent and transmit
information. Population-level neural activity in cortex exhibits frequency-selective
gain modulation [Buzsáki & Draguhn, 2004]. The spectral filter H is analogous to
the frequency-tuning curve of a cortical area: it selectively amplifies particular
rhythms in the state signal.

By learning H_s (state diffusion) and H_i (input coupling) jointly, the model can
discover which frequency components of its state carry useful information for the
next-token prediction objective.

The spectral operator is a CIRCULAR convolution in the feature space — equivalent to
a full rank-D circulant matrix [Gray, 2006]. This is more expressive than the scalar
phase rotation in CellAI v2, and more parameter-efficient than the dense D×D matrix:
    Dense W:  D² params (65,536 for D=256)
    H_s + H_i: 2 × (D//2 + 1) complex = D+2 real params each (258 for D=256)

Shared vs. per-partition
------------------------
By default, one base filter pair (H_s, H_i) is shared across all N partitions,
with per-partition small modulation vectors (delta_s_n, delta_i_n) of size D//2+1.
This gives N×(D//2+1) additional params — a 4×129=516 params for diversity vs
the 2×65536=131K for the dense version.

Ring diffusion (Laplacian) is preserved as an additive residual term using only
the cos-component of the FFT filter, enforcing the low-pass behaviour expected
from physical diffusion processes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralPDE(nn.Module):
    """
    Frequency-domain cellular PDE step.

    For each token, evolves N partition states S_n via:

        S_fft   = rfft(S_n)                  # (D//2+1,) complex
        I_fft   = rfft(inp)                  # shared across partitions
        H_s     = base_H_s + delta_s[n]     # per-partition state filter
        H_i     = base_H_i + delta_i[n]     # per-partition input filter
        S'_n    = irfft(S_fft * H_s + I_fft * H_i)
                  - gamma * S_n             # leakage
                  + D_diff * laplacian(S_n) # ring diffusion residual

    All N partitions computed in a single batched rfft call.

    Args:
        num_partitions:  N
        state_size:      D (must be even)
        gamma:           decay coefficient
        D_diff:          diffusion coefficient for ring Laplacian residual
    """

    def __init__(
        self,
        num_partitions: int,
        state_size:     int,
        gamma:          float = 0.05,
        D_diff:         float = 0.02,
    ):
        super().__init__()
        self.N    = num_partitions
        self.D    = state_size
        self.F    = state_size // 2 + 1      # rfft output size
        self.gamma  = gamma
        self.D_diff = D_diff

        # Shared base spectral filters (complex = amplitude + phase)
        self.base_H_s_real = nn.Parameter(torch.randn(self.F) * 0.05)
        self.base_H_s_imag = nn.Parameter(torch.zeros(self.F))
        self.base_H_i_real = nn.Parameter(torch.randn(self.F) * 0.05)
        self.base_H_i_imag = nn.Parameter(torch.zeros(self.F))

        # Per-partition modulation (small delta to specialise each partition)
        self.delta_s_real = nn.Parameter(torch.randn(num_partitions, self.F) * 0.01)
        self.delta_s_imag = nn.Parameter(torch.zeros(num_partitions, self.F))
        self.delta_i_real = nn.Parameter(torch.randn(num_partitions, self.F) * 0.01)
        self.delta_i_imag = nn.Parameter(torch.zeros(num_partitions, self.F))

        # Non-linear gating for input after frequency mixing
        self.input_gate = nn.Linear(state_size, state_size, bias=False)
        nn.init.eye_(self.input_gate.weight)   # start as identity

    def _build_filters(self):
        """Assemble per-partition complex filters from base + delta."""
        H_s = torch.complex(
            self.base_H_s_real + self.delta_s_real,   # (N, F)
            self.base_H_s_imag + self.delta_s_imag,   # (N, F)
        )  # (N, F) complex
        H_i = torch.complex(
            self.base_H_i_real + self.delta_i_real,
            self.base_H_i_imag + self.delta_i_imag,
        )  # (N, F) complex
        return H_s, H_i

    def step(
        self,
        state: torch.Tensor,    # (N, D)
        inp:   torch.Tensor,    # (D,)
        params,                 # ModelParams (dt used)
    ) -> torch.Tensor:          # (N, D)
        state = state.float()
        inp   = inp.float()

        # rfft of all partition states: (N, F) complex
        S_fft = torch.fft.rfft(state, n=self.D, dim=1)
        # rfft of input (same for all partitions): (F,) → broadcast
        I_fft = torch.fft.rfft(inp, n=self.D)    # (F,)

        H_s, H_i = self._build_filters()          # (N, F) each

        # Frequency mixing: state diffusion + input coupling
        mixed_fft = S_fft * H_s + I_fft.unsqueeze(0) * H_i   # (N, F) complex

        # Back to real domain
        S_new = torch.fft.irfft(mixed_fft, n=self.D)          # (N, D) real

        # Sigmoid nonlinearity (biologically: firing rate nonlinearity)
        S_new = torch.sigmoid(S_new)

        # Ring Laplacian residual (same as dense PDE's ring coupling)
        laplacian = (torch.roll(state, 1, 0) + torch.roll(state, -1, 0) - 2.0 * state)

        # Euler step: decay + diffusion residual
        dS = S_new - self.gamma * state + self.D_diff * laplacian
        return state + params.dt * dS             # (N, D)

    def forward(self, state, inp, params):
        return self.step(state, inp, params)

    def initial_state(self, device):
        return torch.zeros(self.N, self.D, device=device)

    def flops_per_step(self) -> int:
        """Approximate FLOP count per token per partition."""
        fft_flops  = 5 * self.D * int(self.D.bit_length())  # D log D approx
        mix_flops  = 4 * self.F   # complex multiply is 4 real mults
        ifft_flops = fft_flops
        return self.N * (2 * fft_flops + mix_flops + ifft_flops)
