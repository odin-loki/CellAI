"""
Run harness for architecture search (Wave 1).

Writes meta.json beside checkpoints, tracks best+last, atomic JSON saves.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_REPO = Path(__file__).resolve().parents[1]
OUT_DIR = _REPO / "data" / "local" / "arch_search"


@dataclass
class RunMeta:
    experiment_id: str
    hypothesis: str
    protocol_id: str
    protocol_hash: str
    regime: str
    git_hash: str = ""
    started_at: str = ""
    finished_at: str = ""
    device: str = ""
    gpu_name: str = ""
    train_steps: int = 0
    lr: float = 0.0
    resume_from: Optional[str] = None
    headline_metric: str = "text_code_avg_nll"
    headline_value: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.git_hash:
            self.git_hash = get_git_hash()
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.device = "cpu"


def get_git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def save_results_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def write_meta_json(meta: RunMeta, ckpt_stem: str) -> Path:
    path = OUT_DIR / f"{ckpt_stem}.meta.json"
    meta.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(meta), fh, indent=2, default=str)
    return path


class CheckpointTracker:
    """Save last checkpoint every call; promote best on improved headline metric."""

    def __init__(
        self,
        model: torch.nn.Module,
        exp_id: str,
        headline_key: str = "text_code_avg_nll",
        lower_is_better: bool = True,
    ):
        self.model = model
        self.exp_id = exp_id
        self.headline_key = headline_key
        self.lower_is_better = lower_is_better
        self.best_value: Optional[float] = None
        self.best_path = OUT_DIR / f"{exp_id}_best.pt"
        self.last_path = OUT_DIR / f"{exp_id}_last.pt"

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.lower_is_better:
            return value < self.best_value
        return value > self.best_value

    def save(
        self,
        extra: Optional[Dict[str, Any]] = None,
        headline_value: Optional[float] = None,
        tag: str = "last",
    ) -> Path:
        payload = {"state_dict": self.model.state_dict()}
        if extra:
            payload.update(extra)
        if tag == "best" or (headline_value is not None and self._is_better(headline_value)):
            if headline_value is not None:
                self.best_value = headline_value
            torch.save(payload, self.best_path)
            return self.best_path
        torch.save(payload, self.last_path)
        return self.last_path
