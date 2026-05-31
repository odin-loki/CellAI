"""models.software.trad_code — minimal software head."""
from __future__ import annotations
from typing import Optional
import torch.nn as nn
from cellai_core.base  import ModelParams
from models.base_model import CellAIModel

class TradSoftwareModel(CellAIModel):
    MODEL_TYPE = "software_trad"
    def __init__(self, cell_system, params: Optional[ModelParams] = None):
        super().__init__(cell_system, params)
        D = self.params.state_size
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU()).to(self.device)
    def _domain_forward(self, cell_out):
        return self.head(cell_out)
