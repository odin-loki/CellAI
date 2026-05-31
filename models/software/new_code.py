"""
models.software.new_code
========================
Software domain model.

Architecture (on top of cellular backbone):
    cell_out (D,)
        ↓  Split into 4 "artifact" channels (syntax, execution, memory, pattern)
           Each channel: Linear(D/4, D/4) → GELU
        ↓  Dependency-aware cross-channel attention (4×4 attention over channels)
        ↓  Merge channels → (D,) via concat + Linear
        ↓  (D,)

Honest description:
- "Artifact channels" are learned linear projections, not actual code parsers
- "Dependency attention" is scaled dot-product attention over 4 channel vectors
- Output is vocabulary-decoded cellular state, not executable code
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base  import ModelParams
from models.base_model import CellAIModel


N_CHANNELS = 4   # syntax, execution, memory, pattern


class ArtifactChannels(nn.Module):
    """Split D-dim state into N_CHANNELS subspaces, each processed independently."""

    def __init__(self, state_size: int, n_channels: int = N_CHANNELS):
        super().__init__()
        self.ch_dim = state_size // n_channels
        self.n_channels = n_channels
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(self.ch_dim, self.ch_dim), nn.GELU())
            for _ in range(n_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (D,) → (n_channels, ch_dim)"""
        chunks = x.reshape(self.n_channels, self.ch_dim)     # (C, ch_dim)
        return torch.stack([self.projs[i](chunks[i]) for i in range(self.n_channels)])


class ChannelAttention(nn.Module):
    """Cross-channel attention: 4 channels attend to each other."""

    def __init__(self, ch_dim: int):
        super().__init__()
        self.scale = ch_dim ** -0.5
        self.qkv = nn.Linear(ch_dim, 3 * ch_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (C, ch_dim) → (C, ch_dim)"""
        q, k, v = self.qkv(x).chunk(3, dim=-1)              # each (C, ch_dim)
        attn = F.softmax(q @ k.t() * self.scale, dim=-1)    # (C, C)
        return attn @ v                                       # (C, ch_dim)


class SoftwareHead(nn.Module):
    def __init__(self, state_size: int, n_channels: int = N_CHANNELS):
        super().__init__()
        self.channels = ArtifactChannels(state_size, n_channels)
        self.ch_attn  = ChannelAttention(state_size // n_channels)
        self.merge    = nn.Linear(state_size, state_size)
        self.norm     = nn.LayerNorm(state_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch   = self.channels(x)                              # (C, ch_dim)
        ch   = ch + self.ch_attn(ch)                         # residual
        flat = ch.reshape(-1)                                 # (D,)
        return self.norm(self.merge(flat) + x)


class NewSoftwareModel(CellAIModel):
    """Cell AI software model with artifact-channel decomposition."""
    MODEL_TYPE = "software"

    def __init__(self, cell_system, params: Optional[ModelParams] = None):
        super().__init__(cell_system, params)
        # Ensure state_size divisible by N_CHANNELS
        D = self.params.state_size
        if D % N_CHANNELS != 0:
            raise ValueError(f"state_size ({D}) must be divisible by {N_CHANNELS}")
        self.head = SoftwareHead(D).to(self.device)

    def _domain_forward(self, cell_out: torch.Tensor) -> torch.Tensor:
        return self.head(cell_out)
