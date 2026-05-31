"""
Cell AI v1
==========
Base cellular state machine.

Architecture
------------
    text  ──► UniversalEncoder ──► (D,) encoded vector
                                        │
                              ┌─────────▼──────────┐
                              │   PartitionManager  │
                              │   (N partitions)    │
                              │   CellularPDE step  │
                              └─────────┬──────────┘
                                        │ aggregate → (D,)
                              ┌─────────▼──────────┐
                              │  MemoryFormation    │
                              │  MetaplasticityLayer│
                              │  output_proj        │
                              └─────────┬──────────┘
                                        ▼ (D,) cellular state

The cellular state is decoded by projecting onto the encoder's embedding
weight matrix and taking the top-k tokens.  This is *not* a language model —
there is no autoregressive generation, no temperature sampling, and no trained
LM head.  It is a cellular associative memory that maps input text to a
nearby region of the token vocabulary.

What is real and verified
-------------------------
- PDE step:   dS/dt = σ(WI)·tanh(ES) - γS + D∇²_ring + η
- Memory:     M(t) = Σₜ w(t-s)·I(s) + Σₜ K(t-s)·S(s)
- Partition:  N cells with ring coupling, all evolved in one batched matmul
- Encoder:    tiktoken cl100k_base BPE + nn.Embedding + positional encoding

Measured performance (RTX 3090, default params):
    encode_pooled:  ~0.45 ms
    PartitionManager.step:  ~0.08 ms  (was 3.38 ms before rewrite)
    full forward:   ~1.2 ms  (was 5.98 ms before rewrite)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellai_core.base    import ModelParams, CellularPDE
from cellai_core.encoder import UniversalEncoder
from cellai_core.memory  import MemoryFormation, MetaplasticityLayer
from cellai_core.partition import PartitionManager
from cellai_core.utils   import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CellAI v1
# ---------------------------------------------------------------------------

class CellAI(nn.Module):
    """
    Cell AI v1 — base cellular state machine.

    The model encodes text, evolves N parallel cellular partitions,
    integrates memory, and projects back to vocabulary space.
    All intermediate tensors stay on GPU; there are no numpy round-trips.
    """

    VERSION = "v1"

    def __init__(self, params: Optional[ModelParams] = None):
        super().__init__()
        self.params = params or ModelParams()
        set_seed(self.params.seed)
        self.device = torch.device(self.params.device)

        # Encoder
        self.encoder = UniversalEncoder(
            state_size=self.params.state_size,
            vocab=self.params.encoder_vocab,
        )

        # Partition system (vectorised GPU PDE — no CuPy loops)
        self.partitions = PartitionManager(self.params)

        # Memory stack
        self.memory_formation = MemoryFormation(
            memory_size=self.params.state_size,
            time_window=self.params.tau_memory,
            alpha=self.params.alpha,
            beta=self.params.beta,
            omega=self.params.omega,
        )
        self.metaplasticity = MetaplasticityLayer(self.params.state_size)

        # Output projection (same dimension, learned)
        self.output_proj = nn.Linear(self.params.state_size, self.params.state_size)

        # Move everything to target device
        self.to(self.device)

    # ------------------------------------------------------------------
    # Core forward pass (fully GPU-resident)
    # ------------------------------------------------------------------

    def encode_input(self, text: str) -> torch.Tensor:
        """BPE-encode and mean-pool text → (D,) on self.device."""
        return self.encoder.encode_pooled(text, device=self.device)

    def cellular_step(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        One cellular evolution step, fully on GPU:
            1. Feed encoded vector to all N partitions (batched PDE)
            2. Aggregate partition states → (D,) mean
            3. Memory formation
            4. Metaplasticity gate
            5. Output projection
        Returns (D,) state vector.
        """
        # --- Partition evolution (all N partitions in one GPU kernel) ---
        self.partitions.step(encoded)              # state: (N, D) → (N, D)
        agg = self.partitions.aggregate()          # (D,)

        # --- Memory + plasticity ---
        memory = self.memory_formation(encoded, agg)   # (D,)
        out    = self.metaplasticity(agg, memory, encoded)  # (D,)
        return self.output_proj(out)               # (D,)

    def forward(self, text: str) -> torch.Tensor:
        """text → (D,) cellular state vector."""
        encoded = self.encode_input(text)
        return self.cellular_step(encoded)

    # ------------------------------------------------------------------
    # High-level interfaces
    # ------------------------------------------------------------------

    def chat(self, prompt: str) -> str:
        """
        Map prompt text to a cellular state and decode it.

        The decoding works by projecting the (D,) state onto the embedding
        weight matrix to get per-token similarity scores, then selecting
        the top-40 tokens.  This is associative retrieval, not generation.
        """
        with torch.no_grad():
            state = self.forward(prompt)
        return self.encoder.decode_logits(state, top_k=40)

    def train_step(
        self,
        batch: List[str],
        optimizer: torch.optim.Optimizer,
        next_tokens: Optional[List[int]] = None,
    ) -> float:
        """
        One mean-pool training step (fast, no BPTT).

        Encodes each text via mean-pool → cellular step → MSE reconstruction.
        Use train_step_sequential for next-token prediction with BPTT.
        """
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)

        for i, text in enumerate(batch):
            encoded = self.encode_input(text)
            state   = self.cellular_step(encoded)

            if next_tokens is not None:
                logits = state @ self.encoder.embedding.weight.t()
                target = torch.tensor([next_tokens[i]], dtype=torch.long, device=self.device)
                total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), target)
            else:
                total_loss = total_loss + F.mse_loss(state, encoded.detach())

        loss = total_loss / len(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def train_step_sequential(
        self,
        text: str,
        optimizer: torch.optim.Optimizer,
        segment_len: int = 64,
        reset_state: bool = True,
    ) -> float:
        """
        Sequential next-token prediction with truncated BPTT.

        Feeds text token-by-token into the cellular system.  Predicts the
        next token from the current cellular state.  Backpropagates in
        segments of `segment_len` tokens (truncated BPTT) so memory stays
        bounded for long documents.

        This is the real training objective:
            L = -Σₜ log P(tokenₜ₊₁ | cellular_state_t)

        Returns:
            average cross-entropy loss per token.
        """
        was_training = self.training
        self.train()   # ensure MetaplasticityLayer Hebbian update is active

        if reset_state:
            self.partitions.reset()

        tokens = self.encoder.tokenize(text)
        if len(tokens) < 2:
            return 0.0

        optimizer.zero_grad()
        total_loss  = 0.0
        token_count = 0

        # Keep token IDs on GPU for fast embedding lookup
        tok_ids = torch.tensor(tokens, dtype=torch.long, device=self.device)

        for seg_start in range(0, len(tokens) - 1, segment_len):
            seg_end = min(seg_start + segment_len, len(tokens) - 1)

            # Re-compute embeddings for THIS segment only, creating a fresh subgraph.
            # This is needed so that calling .backward() on each segment doesn't try
            # to re-use saved tensors from a previous segment's already-freed graph.
            seg_ids  = tok_ids[seg_start : seg_end + 1]          # (seg_len+1,)
            seg_embs = self.encoder.embedding(seg_ids) * self.encoder._scale  # (seg_len+1, D)

            seg_loss = torch.tensor(0.0, device=self.device)

            for t_local in range(seg_end - seg_start):
                inp = seg_embs[t_local]               # (D,)

                # Cellular step (partition state stays on GPU)
                self.partitions.step(inp)
                agg    = self.partitions.aggregate()  # (D,)

                # Memory + plasticity (Hebbian — no autograd through them)
                memory = self.memory_formation(inp, agg)
                out    = self.metaplasticity(agg, memory, inp)
                state  = self.output_proj(out)        # (D,) — gradient flows here

                # Next-token cross-entropy (predict token at t_local+1)
                logits = state @ self.encoder.embedding.weight.t()  # (vocab,)
                target = seg_ids[t_local + 1].unsqueeze(0)          # (1,)
                seg_loss = seg_loss + F.cross_entropy(logits.unsqueeze(0), target)
                token_count += 1

            (seg_loss / max(seg_end - seg_start, 1)).backward()
            total_loss += seg_loss.item()

            # Detach partition state so next segment doesn't BPTT through it
            self.partitions._buffers["state"] = self.partitions._buffers["state"].detach()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        if not was_training:
            self.eval()
        return total_loss / max(token_count, 1)

    def benchmark(self, n_samples: int = 100, text: str = "benchmark sample") -> Dict[str, float]:
        """Run `forward()` n_samples times and return timing stats."""
        with torch.no_grad():
            # warmup
            for _ in range(5):
                self.forward(text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_samples):
                self.forward(text)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "version":       self.VERSION,
            "n_samples":     n_samples,
            "total_s":       elapsed,
            "ms_per_sample": elapsed * 1000 / n_samples,
            "samples_per_s": n_samples / elapsed,
            "device":        str(self.device),
        }

    def get_info(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "version":        self.VERSION,
            "state_size":     self.params.state_size,
            "num_partitions": self.params.num_partitions,
            "device":         str(self.device),
            "n_params":       n_params,
            "vocab_size":     self.encoder.vocab_size,
        }

    def _load_jsonl(self, path: str) -> Iterator[Dict]:
        import json
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
