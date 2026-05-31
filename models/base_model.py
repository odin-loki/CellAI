"""
models.base_model
=================
Abstract base for all Cell AI domain models.

Every domain model:
    - Takes a cell_system (CellAI or CellAIv2) as backbone
    - Shares the backbone's Universal BPE encoder
    - Exposes chat() / train() / benchmark() / get_info()

Architecture:
    text ──► backbone.encode_input() ──► (D,)
                                          │
                           domain-specific layers
                                          │
                           ──► backbone.encoder.decode_logits() ──► text

The decode step is *associative retrieval* (nearest vocabulary items),
not autoregressive generation.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base    import ModelParams
from cellai_core.utils   import set_seed


class CellAIModel(ABC):
    """
    Abstract base for Cell AI domain models.

    Args:
        cell_system:  a CellAI or CellAIv2 instance (the backbone)
        params:       ModelParams — defaults to cell_system.params
    """

    MODEL_TYPE: str = "base"

    def __init__(self, cell_system, params: Optional[ModelParams] = None):
        self.cell    = cell_system
        self.params  = params or cell_system.params
        self.device  = cell_system.device
        self.encoder = cell_system.encoder       # shared, not copied

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _domain_forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Apply domain-specific layers to the (D,) encoded vector.
        Must return a (D,) tensor on self.device.
        """

    # ------------------------------------------------------------------
    # Shared high-level pipeline
    # ------------------------------------------------------------------

    def _full_forward(self, text: str) -> torch.Tensor:
        """Encode → backbone cellular step → domain layers → (D,)."""
        encoded   = self.cell.encode_input(text)
        cell_out  = self.cell.cellular_step(encoded)
        return self._domain_forward(cell_out)

    def chat(self, prompt: str) -> str:
        """
        Produce a text response via:
            prompt → cellular processing → domain layers → vocabulary lookup
        """
        with torch.no_grad():
            state = self._full_forward(prompt)
        return self.encoder.decode_logits(state, top_k=40)

    def process(self, text: str) -> torch.Tensor:
        """Return raw (D,) state vector (for chaining / analysis)."""
        with torch.no_grad():
            return self._full_forward(text)

    def benchmark(self, n_samples: int = 100, text: str = "benchmark") -> Dict[str, float]:
        """Measure throughput of the full chat() path."""
        with torch.no_grad():
            for _ in range(5):
                self._full_forward(text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_samples):
                self._full_forward(text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "model":         self.MODEL_TYPE,
            "cell_version":  self.cell.VERSION,
            "n_samples":     n_samples,
            "ms_per_sample": elapsed * 1000 / n_samples,
            "samples_per_s": n_samples / elapsed,
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "model":         self.MODEL_TYPE,
            "cell_version":  self.cell.VERSION,
            "state_size":    self.params.state_size,
            "device":        str(self.device),
        }

    def train(
        self,
        data_path: str,
        dry_run: bool = False,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Minimal training loop.  Domain models should override this with
        task-specific objectives.  Base implementation uses MSE reconstruction.
        """
        if dry_run:
            return {"status": "dry_run_ok", "model": self.MODEL_TYPE}
        epochs = epochs or self.params.max_epochs
        optimizer = torch.optim.AdamW(
            [p for p in self.cell.parameters() if p.requires_grad],
            lr=self.params.learning_rate,
        )
        total_loss, steps = 0.0, 0
        for _ in range(epochs):
            for record in self._load_jsonl(data_path):
                text = record.get("text", record.get("problem", str(record)))
                loss = self.cell.train_step([text], optimizer)
                total_loss += loss
                steps += 1
        return {"epochs": epochs, "steps": steps, "avg_loss": total_loss / max(steps, 1)}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _load_jsonl(self, path: str) -> Iterator[Dict]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
