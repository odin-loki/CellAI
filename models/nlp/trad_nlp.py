"""
models.nlp.trad_nlp
===================
Minimal NLP model: single linear projection + LayerNorm.
Useful as a lightweight baseline.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from cellai_core.base  import ModelParams
from models.base_model import CellAIModel


class TradNLPModel(CellAIModel):
    """Lightweight NLP head: LayerNorm + Linear (no attention)."""
    MODEL_TYPE = "nlp_trad"

    def __init__(self, cell_system, params: Optional[ModelParams] = None):
        super().__init__(cell_system, params)
        D = self.params.state_size
        self.head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
        ).to(self.device)

    def _domain_forward(self, cell_out: torch.Tensor) -> torch.Tensor:
        return self.head(cell_out)
