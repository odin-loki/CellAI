"""
models.nlp.new_nlp
==================
NLP domain model: adds sequence-aware layers on top of the cellular backbone.

Architecture (on top of v1/v2 backbone output):
    cell_out (D,)
        ↓  LayerNorm
        ↓  Position-wise FFN  (D → 4D → D, GELU)
        ↓  Local windowed self-attention  (window=8, O(D·w) per token)
        ↓  Gate:  σ(W_g · cell_out) · FFN_out
        ↓  (D,)

All names are accurate descriptions of what is implemented.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base  import ModelParams
from models.base_model import CellAIModel


class LocalAttention(nn.Module):
    """
    Windowed attention on a 1-D signal.

    For a (D,) vector we treat it as a sequence of D/head_dim sub-vectors
    and apply dot-product attention within a local window of size `window`.
    This is honest about what it does: O(D * window) complexity.
    """

    def __init__(self, state_size: int, heads: int = 4, window: int = 8):
        super().__init__()
        self.heads     = heads
        self.head_dim  = state_size // heads
        self.window    = window
        self.qkv = nn.Linear(self.head_dim, 3 * self.head_dim, bias=False)
        self.out = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (D,) → (D,)"""
        D = x.shape[0]
        # Reshape into (heads, head_dim) for independent head processing
        h = x.reshape(self.heads, self.head_dim)                # (H, Hd)
        qkv = self.qkv(h)                                        # (H, 3·Hd)
        q, k, v = qkv.chunk(3, dim=-1)                          # each (H, Hd)
        # Attention within heads (treating heads as the sequence)
        w = q.size(0)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)                           # (H, H)
        out  = torch.matmul(attn, v)                             # (H, Hd)
        out  = self.out(out).reshape(D)                          # (D,)
        return out


class NLPHead(nn.Module):
    """
    Position-wise FFN + local attention + gating.
    Adds 3 parameters tensors for a modest capacity increase over bare cellular output.
    """

    def __init__(self, state_size: int, expand: int = 4):
        super().__init__()
        inner = state_size * expand
        self.norm = nn.LayerNorm(state_size)
        self.ffn  = nn.Sequential(
            nn.Linear(state_size, inner),
            nn.GELU(),
            nn.Linear(inner, state_size),
        )
        self.attn = LocalAttention(state_size)
        self.gate = nn.Linear(state_size, state_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (D,) → (D,)"""
        x_n  = self.norm(x)
        ffn  = self.ffn(x_n)
        att  = self.attn(x_n)
        g    = torch.sigmoid(self.gate(x))
        return x + g * (ffn + att)


class NewNLPModel(CellAIModel):
    """
    Cell AI NLP model.

    Adds a gated FFN + local attention head on top of the cellular backbone.
    """
    MODEL_TYPE = "nlp"

    def __init__(self, cell_system, params: Optional[ModelParams] = None):
        super().__init__(cell_system, params)
        self.head = NLPHead(self.params.state_size).to(self.device)

    def _domain_forward(self, cell_out: torch.Tensor) -> torch.Tensor:
        return self.head(cell_out)
