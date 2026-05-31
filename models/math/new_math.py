"""
models.math.new_math
====================
Math domain model.

Architecture (on top of cellular backbone):
    cell_out (D,)
        ↓  Tensor Product Representation layer
           Maps (D,) → roles (R,) and fillers (F,), outer-product → (R·F,) → proj (D,)
        ↓  Symbolic decision layer
           Classifies into operation type {arithmetic, algebra, calculus, ...}
           then applies an operation-specific projection
        ↓  (D,)

Honest description:
- TPR approximates symbolic structure via learned role/filler decomposition
- The operation classifier is a simple 8-way linear softmax head
- Output is a (D,) vector decoded to tokens (not a symbolic math engine)
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base  import ModelParams
from models.base_model import CellAIModel


N_OPS = 8   # arithmetic, algebra, calculus, number_theory, geometry, stats, logic, other


class TPRLayer(nn.Module):
    """
    Tensor Product Representation approximation.

    Learns separate role (R-dim) and filler (F-dim) projections,
    forms their outer product (R×F), and projects back to D.

    Unlike a true TPR, this shares weights across the outer product
    (no per-cell binding), which is a reasonable approximation for
    fixed-length state vectors.
    """

    def __init__(self, state_size: int, roles: int = 16, fillers: int = 16):
        super().__init__()
        self.role_proj   = nn.Linear(state_size, roles)
        self.filler_proj = nn.Linear(state_size, fillers)
        self.out_proj    = nn.Linear(roles * fillers, state_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.role_proj(x)           # (R,)
        f = self.filler_proj(x)         # (F,)
        tp = torch.outer(r, f).reshape(-1)   # (R·F,)
        return self.out_proj(tp)        # (D,)


class OpClassifier(nn.Module):
    """
    8-class operation classifier + operation-conditioned projection.
    Each operation class has its own learned (D,D) projection.
    """

    def __init__(self, state_size: int, n_ops: int = N_OPS):
        super().__init__()
        self.classifier = nn.Linear(state_size, n_ops)
        # Stack of n_ops projection matrices as one (n_ops, D, D) tensor
        self.op_projs = nn.Parameter(
            torch.randn(n_ops, state_size, state_size) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)              # (n_ops,)
        weights = F.softmax(logits, dim=0)       # (n_ops,) soft gate
        # Weighted sum of op projections: (D,D) from (n_ops, D, D)
        proj = torch.einsum("n, n d e -> d e", weights, self.op_projs)  # (D,D)
        return F.gelu(proj @ x)                  # (D,)


class NewMathModel(CellAIModel):
    """Cell AI math model with TPR decomposition and operation routing."""
    MODEL_TYPE = "math"

    def __init__(self, cell_system, params: Optional[ModelParams] = None):
        super().__init__(cell_system, params)
        D = self.params.state_size
        self.norm   = nn.LayerNorm(D).to(self.device)
        self.tpr    = TPRLayer(D).to(self.device)
        self.ops    = OpClassifier(D).to(self.device)
        self.output = nn.Linear(D, D).to(self.device)

    def _domain_forward(self, cell_out: torch.Tensor) -> torch.Tensor:
        x = self.norm(cell_out)
        x = x + self.tpr(x)
        x = x + self.ops(x)
        return self.output(x)
