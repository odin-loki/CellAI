"""
Eval protocol for architecture search (Wave 1).

Single source of truth for burn length, BPTT detach cadence, sample counts,
and headline metrics (text+code vs macro).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "config" / "eval_protocol_v1.json"

TEXT_CODE_MODS = ("text", "code")


@dataclass
class EvalConfig:
    """Evaluation settings for one training regime."""

    protocol_id: str = "arch_eval_v1"
    version: int = 1
    regime: str = "reset"  # "reset" | "continuous"
    seed: int = 42
    tokenizer: str = "cl100k_base"
    n_per_modality: int = 50
    max_tokens: int = 256
    warm_tokens: int = 0
    detach_every: Optional[int] = None
    stream_burn_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def protocol_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def n_total(self) -> int:
        return self.n_per_modality * 3


def load_eval_protocol(
    regime: str = "reset",
    path: Optional[Path] = None,
) -> EvalConfig:
    """Load protocol JSON and return config for ``regime``."""
    p = path or _DEFAULT_PROTOCOL
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    block = raw[regime]
    return EvalConfig(
        protocol_id=raw["protocol_id"],
        version=raw["version"],
        regime=regime,
        seed=raw.get("seed", 42),
        tokenizer=raw.get("tokenizer", "cl100k_base"),
        n_per_modality=block["n_per_modality"],
        max_tokens=block.get("max_tokens", 256),
        warm_tokens=block.get("warm_tokens", 0),
        detach_every=block.get("detach_every"),
        stream_burn_tokens=block.get("stream_burn_tokens"),
    )


def enrich_ppl_metrics(ppl: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add text+code headline metrics alongside macro (continuity with E0–E26).

    Modifies and returns ``ppl`` in place.
    """
    tc_nlls = []
    for mod in TEXT_CODE_MODS:
        stats = ppl.get(mod)
        if isinstance(stats, dict) and "avg_nll" in stats:
            tc_nlls.append(stats["avg_nll"])
    if tc_nlls:
        avg = sum(tc_nlls) / len(tc_nlls)
        ppl["text_code_avg_nll"] = avg
        ppl["text_code_ppl"] = math.exp(min(avg, 30.0))
    return ppl


def attach_protocol(ppl: Dict[str, Any], cfg: EvalConfig) -> Dict[str, Any]:
    """Embed protocol metadata on a perplexity result block."""
    ppl = enrich_ppl_metrics(ppl)
    ppl["protocol"] = cfg.to_dict()
    ppl["protocol_hash"] = cfg.protocol_hash
    return ppl
