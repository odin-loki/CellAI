"""
models.multimodal.multimodal
============================
Multimodal routing model.

What this does (honest description)
-------------------------------------
Routes the input through one of three modality-specialised heads based on
a learned classifier, then blends outputs with learned mixture weights.

Modality detection is heuristic + learned:
    - A lightweight linear classifier scores the encoded state for
      text / code / math likelihood
    - Each modality has its own 2-layer MLP head
    - Final output = Σ softmax(classifier) * head_output(state)

This is NOT:
    - A vision-language model
    - A model that processes images or audio
    - A model with cross-modal alignment training

The "multimodal" aspect is that the same cellular backbone can handle
text, code, and mathematical notation via a single shared encoder
and three specialised projection heads.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base  import ModelParams
from models.base_model import CellAIModel


MODALITIES = ["text", "code", "math"]


class ModalityHead(nn.Module):
    """Two-layer MLP for one modality."""

    def __init__(self, state_size: int):
        super().__init__()
        inner = state_size * 2
        self.net = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, inner),
            nn.GELU(),
            nn.Linear(inner, state_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalModel(CellAIModel):
    """
    Cell AI multimodal model with modality routing.

    Three modalities supported: text, code, math.
    Routing is soft (mixture of heads, not hard switching).
    """
    MODEL_TYPE = "multimodal"

    def __init__(self, cell_system, params: Optional[ModelParams] = None):
        super().__init__(cell_system, params)
        D = self.params.state_size
        n = len(MODALITIES)
        self.router = nn.Linear(D, n).to(self.device)
        self.heads  = nn.ModuleList([ModalityHead(D) for _ in range(n)]).to(self.device)

    def _domain_forward(self, cell_out: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.router(cell_out), dim=0)       # (n,)
        head_outs = torch.stack(
            [h(cell_out) for h in self.heads], dim=0
        )                                                         # (n, D)
        return (weights.unsqueeze(1) * head_outs).sum(dim=0)    # (D,)

    def detect_modality(self, prompt: str) -> str:
        """Infer the most likely modality for this prompt."""
        with torch.no_grad():
            enc = self.cell.encode_input(prompt)
            w   = F.softmax(self.router(enc), dim=0)
        return MODALITIES[w.argmax().item()]
