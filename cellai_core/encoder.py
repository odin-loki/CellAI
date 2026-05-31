"""
cellai_core.encoder
===================
Universal BPE encoder for all Cell AI models.

Pipeline:
    text / code / math
        ↓  tiktoken cl100k_base  (byte-level BPE, 100 277 tokens)
        ↓  nn.Embedding(vocab_size, D)  +  sinusoidal PositionalEncoding
        ↓  mean-pool across tokens  →  (D,)  cellular state vector

This is the only input stage: NLP, math, code and multimodal text all share it.
The vocab is the same tokeniser used by GPT-3.5 / GPT-4; the embedding weights
are randomly initialised and trained from scratch (or fine-tuned).
"""
from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Positional encoding (Vaswani et al., 2017)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — fixed, not learned.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq, D)  or  (batch, seq, D)
        if x.dim() == 2:
            x = x + self.pe[0, : x.size(0)]
        else:
            x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Universal encoder
# ---------------------------------------------------------------------------

class UniversalEncoder(nn.Module):
    """
    Two-stage BPE encoder.

    Stage 1 — tiktoken tokenisation → integer IDs
    Stage 2 — nn.Embedding(vocab_size, D) + PositionalEncoding → (seq, D)

    Calling encode_pooled(text) returns a single (D,) vector suitable as
    the input signal for the cellular PDE.

    Args:
        state_size:  D  — embedding and cellular state dimension
        vocab:       tiktoken encoding name ("cl100k_base" = 100 277 tokens)
        max_len:     maximum supported sequence length
        dropout:     positional-encoding dropout (only active in training mode)
    """

    def __init__(
        self,
        state_size: int,
        vocab: str = "cl100k_base",
        max_len: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required: pip install tiktoken")

        self.D = state_size
        self._tok = tiktoken.get_encoding(vocab)
        self.vocab_size = self._tok.n_vocab

        self.embedding   = nn.Embedding(self.vocab_size, state_size, padding_idx=0)
        self.pos_enc     = PositionalEncoding(state_size, max_len=max_len, dropout=dropout)
        self._scale      = math.sqrt(state_size)

    # ------------------------------------------------------------------
    # Tokenisation helpers (CPU, no gradients)
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def detokenize(self, ids: Union[List[int], torch.Tensor]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        valid = [i for i in ids if 0 < i < self.vocab_size]
        try:
            return self._tok.decode(valid)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Encoding (GPU-resident)
    # ------------------------------------------------------------------

    def encode(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode text → (seq_len, D) tensor."""
        ids = torch.tensor(self._tok.encode(text), dtype=torch.long)
        if device is not None:
            ids = ids.to(device)
        emb = self.embedding(ids) * self._scale        # (seq, D)
        return self.pos_enc(emb)

    def encode_pooled(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode text and mean-pool → (D,) vector."""
        return self.encode(text, device=device).mean(dim=0)

    def encode_batch(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        pad_to: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode a batch of strings → (batch, seq_len, D)."""
        seqs = [self._tok.encode(t) for t in texts]
        max_seq = pad_to if pad_to else max(len(s) for s in seqs)
        padded  = [s[:max_seq] + [0] * max(0, max_seq - len(s)) for s in seqs]
        ids = torch.tensor(padded, dtype=torch.long)
        if device is not None:
            ids = ids.to(device)
        emb = self.embedding(ids) * self._scale        # (B, seq, D)
        return self.pos_enc(emb)

    def encode_batch_pooled(
        self, texts: List[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Encode batch and pool → (batch, D)."""
        return self.encode_batch(texts, device=device).mean(dim=1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, seq) or (seq,) → (B, seq, D) or (seq, D)."""
        sq = ids.dim() == 1
        if sq:
            ids = ids.unsqueeze(0)
        out = self.pos_enc(self.embedding(ids) * self._scale)
        return out.squeeze(0) if sq else out

    # ------------------------------------------------------------------
    # Decoding helper: logits → text
    # ------------------------------------------------------------------

    def decode_logits(self, logits: torch.Tensor, top_k: int = 40) -> str:
        """
        Given a (D,) state vector, project through the embedding matrix,
        take top-k token IDs, and decode them back to a string.

        This is NOT a language model — there is no softmax temperature,
        no autoregressive generation, and no trained LM head.
        It maps a cellular state to the nearest vocabulary items.
        """
        with torch.no_grad():
            scores = logits @ self.embedding.weight.t()  # (vocab_size,)
            ids    = scores.topk(top_k).indices.tolist()
        return self.detokenize(ids)
