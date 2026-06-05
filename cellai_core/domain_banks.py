"""
Input-side domain routing and PDE filter banks (Wave 3 / v3.2).

Hypothesis: cellular-state routing is washed out by diffusion; lightweight
routing from the *token embedding* can specialise slow spectral dynamics
per modality without relying on partition state softmax.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.spectral_pde import SpectralPDE


class InputDomainRouter(nn.Module):
    """Map token embedding (D,) → domain mixture weights (K,)."""

    def __init__(self, state_size: int, num_domains: int = 3, temperature: float = 1.0):
        super().__init__()
        self.num_domains = num_domains
        self.temperature = temperature
        self.proj = nn.Linear(state_size, num_domains, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = self.proj(encoded.float()) / self.temperature
        return F.softmax(logits, dim=-1)


class BankedSpectralPDE(SpectralPDE):
    """
    SpectralPDE with K domain-specific delta banks mixed by input-side weights.

    Effective per-partition deltas:
        delta_eff = delta_base + sum_k w_k * bank_delta_k
    """

    def __init__(
        self,
        num_partitions: int,
        state_size: int,
        num_banks: int = 3,
        **kwargs,
    ):
        super().__init__(num_partitions, state_size, **kwargs)
        self.num_banks = num_banks
        F_sz = self.F
        self.bank_delta_s_real = nn.Parameter(
            torch.randn(num_banks, num_partitions, F_sz) * 0.01
        )
        self.bank_delta_s_imag = nn.Parameter(
            torch.zeros(num_banks, num_partitions, F_sz)
        )
        self.bank_delta_i_real = nn.Parameter(
            torch.randn(num_banks, num_partitions, F_sz) * 0.01
        )
        self.bank_delta_i_imag = nn.Parameter(
            torch.zeros(num_banks, num_partitions, F_sz)
        )

    def _build_filters(self, domain_w: Optional[torch.Tensor] = None):
        if domain_w is None:
            return super()._build_filters()

        w = domain_w.float().reshape(-1)
        if w.numel() != self.num_banks:
            return super()._build_filters()

        mix_s_r = torch.einsum("k,knf->nf", w, self.bank_delta_s_real)
        mix_s_i = torch.einsum("k,knf->nf", w, self.bank_delta_s_imag)
        mix_i_r = torch.einsum("k,knf->nf", w, self.bank_delta_i_real)
        mix_i_i = torch.einsum("k,knf->nf", w, self.bank_delta_i_imag)

        H_s = torch.complex(
            self.base_H_s_real + self.delta_s_real + mix_s_r,
            self.base_H_s_imag + self.delta_s_imag + mix_s_i,
        )
        H_i = torch.complex(
            self.base_H_i_real + self.delta_i_real + mix_i_r,
            self.base_H_i_imag + self.delta_i_imag + mix_i_i,
        )
        return H_s, H_i

    def step(
        self,
        state: torch.Tensor,
        inp: torch.Tensor,
        params,
        domain_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = state.float()
        inp = inp.float()

        S_fft = torch.fft.rfft(state, n=self.D, dim=1)
        I_fft = torch.fft.rfft(inp, n=self.D)

        H_s, H_i = self._build_filters(domain_w)
        mixed_fft = S_fft * H_s + I_fft.unsqueeze(0) * H_i
        S_new = torch.fft.irfft(mixed_fft, n=self.D)
        S_new = torch.sigmoid(S_new)

        laplacian = (torch.roll(state, 1, 0) + torch.roll(state, -1, 0) - 2.0 * state)
        dS = S_new - self.gamma * state + self.D_diff * laplacian
        return state + params.dt * dS
