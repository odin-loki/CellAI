"""
Cell AI â€” Full pipeline: data, training, profiling, evaluation.

Stages:
  1. Generate 1 GB of math problems
  2. Download ~1 GB of NLP text (wikitext-103)
  3. Download ~1 GB of code (code_search_net Python)
  4. Train math / NLP / software models (sequential next-token objective)
  5. Fine-tooth-comb: gradient norms, activation stats, loss curves, throughput
  6. Evaluate: chat examples per domain

Run (from repo root):
    .venv\Scripts\python scripts\run_full_pipeline.py > C:\Temp\pipeline.log 2>&1
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# â”€â”€ Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("pipeline")
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = _REPO / "data" / "local"           # ~1 GB per category will go here
CKPT_DIR = DATA_DIR / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

log.info(f"Device: {DEVICE}")
log.info(f"Data dir: {DATA_DIR.resolve()}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 1: Math data generation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def stage_math(target_gb: float = 1.0) -> Path:
    """Generate math problems until we have target_gb of data."""
    math_dir = DATA_DIR / "math"
    math_dir.mkdir(exist_ok=True)
    out_path  = math_dir / "train.jsonl"
    val_path  = math_dir / "val.jsonl"

    # Estimate problems needed: each record â‰ˆ 400 bytes on average
    avg_bytes_per_record = 400
    target_bytes = int(target_gb * 1e9)
    n_problems   = target_bytes // avg_bytes_per_record
    log.info(f"[MATH] Generating {n_problems:,} problems (~{target_gb:.1f} GB target) ...")

    from data.pipelines.math_pipeline import generate
    counts = generate(count=n_problems, output_dir=math_dir, seed=42)

    actual_size = sum(p.stat().st_size for p in math_dir.glob("*.jsonl")) / 1e9
    n_train = sum(1 for _ in open(out_path, encoding="utf-8"))
    log.info(f"[MATH] Done â€” {actual_size:.2f} GB, {n_train:,} train samples, counts={counts}")
    return out_path


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 2: NLP data (wikitext-103)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def stage_nlp(target_gb: float = 1.0) -> Path:
    """Download wikitext-103 and write JSONL until target_gb reached."""
    nlp_dir  = DATA_DIR / "nlp"
    nlp_dir.mkdir(exist_ok=True)
    out_path = nlp_dir / "train.jsonl"

    if out_path.exists() and out_path.stat().st_size > 0.9e9:
        log.info(f"[NLP] Already have {out_path.stat().st_size/1e9:.2f} GB, skipping download.")
        return out_path

    log.info("[NLP] Downloading wikitext-103-raw-v1 from HuggingFace ...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=False)

    count = 0
    bytes_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            if len(text) < 50:
                continue
            record = json.dumps({"text": text, "source": "wikitext103"}, ensure_ascii=False)
            f.write(record + "\n")
            bytes_written += len(record.encode()) + 1
            count += 1

    # If wikitext alone is < target, also pull ag_news + roneneldan/TinyStories
    if bytes_written < target_gb * 1e9:
        log.info("[NLP] wikitext-103 was %.2f GB, augmenting with ag_news + TinyStories ..." % (bytes_written/1e9))
        for aug_source, aug_field in [
            ("ag_news", "text"),
            ("roneneldan/TinyStories", "story"),
        ]:
            if bytes_written >= target_gb * 1e9:
                break
            try:
                ds2 = load_dataset(aug_source, split="train", streaming=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    for item in ds2:
                        if bytes_written >= target_gb * 1e9:
                            break
                        text = item.get(aug_field, item.get("text", "")).strip()
                        if len(text) < 50:
                            continue
                        record = json.dumps({"text": text, "source": aug_source}, ensure_ascii=False)
                        f.write(record + "\n")
                        bytes_written += len(record.encode()) + 1
                        count += 1
            except Exception as e:
                log.warning(f"[NLP] {aug_source} failed ({e}), skipping.")

    actual_gb = out_path.stat().st_size / 1e9
    log.info(f"[NLP] Done â€” {actual_gb:.2f} GB, {count:,} records â†’ {out_path}")
    return out_path


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 3: Code data (code_search_net Python)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def stage_code(target_gb: float = 1.0) -> Path:
    """Download code_search_net Python functions to JSONL."""
    code_dir = DATA_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    out_path = code_dir / "train.jsonl"

    if out_path.exists() and out_path.stat().st_size > 0.9e9:
        log.info(f"[CODE] Already have {out_path.stat().st_size/1e9:.2f} GB, skipping.")
        return out_path

    log.info("[CODE] Downloading code_search_net (Python) from HuggingFace ...")
    from datasets import load_dataset
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)

    count = 0
    bytes_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            code = item.get("func_code_string", item.get("content", "")).strip()
            doc  = item.get("func_documentation_string", "").strip()
            if len(code) < 30:
                continue
            # Combine docstring + code for richer context
            text = (f'"""{doc}"""\n' if doc else "") + code
            text = text[:8000]   # cap at 8K chars to avoid huge tokens
            record = json.dumps({"content": text, "language": "python",
                                 "source": "code_search_net"}, ensure_ascii=False)
            f.write(record + "\n")
            bytes_written += len(record.encode()) + 1
            count += 1
            if bytes_written >= target_gb * 1e9:
                break

    # If code_search_net alone is < target, also pull codeparrot-clean
    if bytes_written < target_gb * 1e9:
        log.info("[CODE] Augmenting with codeparrot-clean ...")
        try:
            ds2 = load_dataset("codeparrot/codeparrot-clean", split="train",
                               streaming=True, trust_remote_code=True)
            with open(out_path, "a", encoding="utf-8") as f:
                for item in ds2:
                    if bytes_written >= target_gb * 1e9:
                        break
                    code = item.get("content", "").strip()
                    if len(code) < 30:
                        continue
                    code = code[:8000]
                    record = json.dumps({"content": code, "language": "python",
                                         "source": "codeparrot"}, ensure_ascii=False)
                    f.write(record + "\n")
                    bytes_written += len(record.encode()) + 1
                    count += 1
        except Exception as e:
            log.warning(f"[CODE] codeparrot failed ({e}), continuing.")

    actual_gb = out_path.stat().st_size / 1e9
    log.info(f"[CODE] Done â€” {actual_gb:.2f} GB, {count:,} records â†’ {out_path}")
    return out_path


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 4: Trainer
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class Trainer:
    """
    Trains a CellAI (v1 or v2) using sequential next-token prediction.

    Honest about what this is:
      - Single-layer cellular state machine, not a transformer or RNN
      - Loss = cross-entropy next-token prediction
      - Backprop through: output_proj, pde.W/E, encoder.embedding
      - Memory and metaplasticity layers update via Hebbian rules (no backprop)
    """

    def __init__(
        self,
        model,                          # CellAI or CellAIv2
        data_path: Path,
        text_field: str = "text",       # or "content" for code
        lr: float = 5e-4,
        max_steps: int = 5_000,
        segment_len: int = 64,          # BPTT segment length
        log_every: int = 200,
        ckpt_dir: Path = CKPT_DIR,
        name: str = "model",
    ):
        self.model       = model
        self.data_path   = data_path
        self.text_field  = text_field
        self.lr          = lr
        self.max_steps   = max_steps
        self.segment_len = segment_len
        self.log_every   = log_every
        self.ckpt_dir    = ckpt_dir
        self.name        = name

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr * 0.1,
        )
        # Metrics tracking
        self.losses: List[float] = []
        self.steps:  List[int]   = []

    def _iter_texts(self):
        """Yield texts from the JSONL file, cycling indefinitely."""
        while True:
            with open(self.data_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        text = rec.get(self.text_field, "")
                        if len(text) >= 50:
                            yield text
                    except Exception:
                        continue

    def train(self) -> Dict:
        """Run training loop. Returns final metrics."""
        log.info(f"[{self.name}] Training for {self.max_steps} steps, lr={self.lr}, segment={self.segment_len}")
        log.info(f"[{self.name}] Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        self.model.train()
        text_iter  = self._iter_texts()
        running_loss = 0.0
        t0 = time.perf_counter()

        for step in range(1, self.max_steps + 1):
            text = next(text_iter)
            loss = self.model.train_step_sequential(
                text,
                self.optimizer,
                segment_len=self.segment_len,
                reset_state=True,
            )
            self.scheduler.step()
            running_loss += loss

            if step % self.log_every == 0:
                avg = running_loss / self.log_every
                elapsed = time.perf_counter() - t0
                lr_now  = self.scheduler.get_last_lr()[0]
                tps     = step / elapsed
                log.info(
                    f"[{self.name}] step={step:>5} loss={avg:.4f}  "
                    f"lr={lr_now:.2e}  {tps:.1f} steps/s"
                )
                self.losses.append(avg)
                self.steps.append(step)
                running_loss = 0.0

        # Save checkpoint
        ckpt = self.ckpt_dir / f"{self.name}.pt"
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps":      self.steps,
            "losses":     self.losses,
        }, ckpt)
        log.info(f"[{self.name}] Checkpoint saved â†’ {ckpt}")

        self.model.eval()
        return {
            "name":       self.name,
            "max_steps":  self.max_steps,
            "final_loss": self.losses[-1] if self.losses else float("nan"),
            "losses":     self.losses,
            "steps":      self.steps,
        }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 5: Fine-tooth-comb analysis
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def analyse_model(model, text_sample: str, name: str) -> Dict:
    """
    Collect gradient norms, activation stats, and throughput for a model.
    """
    log.info(f"[ANALYSE] {name}")

    # â”€â”€ gradient norms after one backward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model.train()
    opt_tmp = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_tmp.zero_grad()
    model.train_step_sequential(text_sample, opt_tmp, segment_len=32, reset_state=True)

    grad_norms = {}
    for pname, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[pname] = param.grad.norm().item()
        else:
            grad_norms[pname] = 0.0
    del opt_tmp
    model.eval()

    # â”€â”€ activation statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model.train()  # needed to track; but with no_grad for pure stats
    act_stats: Dict = {}
    hooks = []

    def make_hook(layer_name):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor):
                t = out.detach().float()
                act_stats[layer_name] = {
                    "mean": t.mean().item(),
                    "std":  t.std().item(),
                    "max":  t.abs().max().item(),
                    "shape": list(t.shape),
                }
        return hook

    for mname, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(make_hook(mname)))

    with torch.no_grad():
        _ = model.forward(text_sample[:200])
    for h in hooks:
        h.remove()
    model.eval()

    # â”€â”€ throughput â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model.eval()
    with torch.no_grad():
        for _ in range(5):   # warmup
            model.forward("hello world")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    N_BENCH = 50
    with torch.no_grad():
        for _ in range(N_BENCH):
            model.forward("hello world")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per = (time.perf_counter() - t0) / N_BENCH * 1000

    # â”€â”€ parameter statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    param_stats = {}
    for pname, param in model.named_parameters():
        t = param.detach().float()
        param_stats[pname] = {
            "numel":  t.numel(),
            "mean":   t.mean().item(),
            "std":    t.std().item(),
            "norm":   t.norm().item(),
        }

    result = {
        "name":           name,
        "ms_per_forward": ms_per,
        "n_params":       sum(p.numel() for p in model.parameters()),
        "grad_norms":     grad_norms,
        "act_stats":      act_stats,
        "param_stats":    param_stats,
    }
    log.info(f"  Forward: {ms_per:.2f} ms  |  n_params={result['n_params']:,}")
    log.info(f"  Gradient norms:")
    for pn, gn in sorted(grad_norms.items(), key=lambda x: -x[1]):
        if gn > 0:
            log.info(f"    {pn:<52} {gn:.4e}")
    dead = sum(1 for gn in grad_norms.values() if gn == 0)
    log.info(f"  Dead params (grad=0): {dead}/{len(grad_norms)}")
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 6: Chat evaluation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

EVAL_PROMPTS = {
    "math":     [
        "Solve for x: 3x + 7 = 22",
        "Find the derivative of xÂ³ + 2x",
        "What is P(X=2) for X~Binomial(10, 0.3)?",
    ],
    "nlp":      [
        "What is machine learning?",
        "Explain the concept of neural networks.",
        "What is the difference between supervised and unsupervised learning?",
    ],
    "software": [
        "def fibonacci(n):",
        "class BinarySearchTree:",
        "import torch; model = ",
    ],
    "multimodal": [
        "Differentiate xÂ² + sin(x)",
        "Write a Python function to sort a list",
        "The French Revolution began in",
    ],
}


def eval_chat(model, domain: str, prompts: List[str], name: str) -> List[Dict]:
    """Run chat on each prompt and collect results."""
    results = []
    log.info(f"[EVAL/{name}] Chat examples:")
    for prompt in prompts:
        with torch.no_grad():
            t0 = time.perf_counter()
            response = model.chat(prompt)
            ms = (time.perf_counter() - t0) * 1000
        log.info(f"  Q: {prompt[:60]!r}")
        log.info(f"  A: {response[:120]!r}  [{ms:.0f}ms]")
        results.append({"prompt": prompt, "response": response, "ms": ms})
    return results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 7: Multimodal model test
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_multimodal(params):
    """Test the multimodal model's routing and generation."""
    from models.registry import get_model
    log.info("[MULTIMODAL] Testing multimodal routing ...")
    mm = get_model("multimodal", version="v2", params=params)

    test_inputs = [
        ("text",     "The theory of relativity was developed by Einstein."),
        ("math",     "Integrate 2x + 3 from 0 to 5."),
        ("code",     "def merge_sort(arr): return sorted(arr)"),
        ("mixed",    "Find the eigenvalues of [[1,2],[3,4]] using Python"),
    ]

    results = {}
    for label, prompt in test_inputs:
        detected = mm.detect_modality(prompt)
        with torch.no_grad():
            resp = mm.chat(prompt)
        results[label] = {"detected": detected, "response": resp[:80]}
        log.info(f"  [{label}] detected={detected}  prompt={prompt[:50]!r}")
        log.info(f"         response={resp[:80]!r}")
    return results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STAGE 8: Loss curve & perplexity
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_perplexity(model, data_path: Path, text_field: str, n_samples: int = 200) -> float:
    """
    Compute approximate perplexity on the first n_samples from data_path.
    PPL = exp(avg cross-entropy loss per token).
    """
    model.eval()
    total_loss  = 0.0
    total_tokens = 0

    with open(data_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec  = json.loads(line)
                text = rec.get(text_field, "")
            except Exception:
                continue
            if len(text) < 50:
                continue

            tokens = model.encoder.tokenize(text)[:256]
            if len(tokens) < 2:
                continue

            model.partitions.reset()
            tok_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            embs    = model.encoder.embedding(tok_ids) * model.encoder._scale

            seg_loss = 0.0
            for t in range(len(tokens) - 1):
                with torch.no_grad():
                    model.partitions.step(embs[t])
                    agg    = model.partitions.aggregate()
                    memory = model.memory_formation(embs[t], agg)
                    out    = model.metaplasticity(agg, memory, embs[t])
                    state  = model.output_proj(out)
                    logits = state @ model.encoder.embedding.weight.t()
                    lp     = F.cross_entropy(logits.unsqueeze(0), tok_ids[t+1].unsqueeze(0))
                    seg_loss += lp.item()

            total_loss   += seg_loss
            total_tokens += len(tokens) - 1

    if total_tokens == 0:
        return float("nan"), float("nan")
    avg_loss = total_loss / total_tokens
    # Cap PPL to avoid overflow for untrained/poorly-trained models
    ppl = math.exp(min(avg_loss, 30.0))   # exp(30) ≈ 10^13
    return avg_loss, ppl


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    from cellai_core.base import ModelParams
    from v1.cell_ai    import CellAI
    from v2.cell_ai_v2 import CellAIv2
    from models.registry import get_model

    params = ModelParams(state_size=256, num_partitions=4, learning_rate=5e-4)

    print("=" * 70)
    print("CELL AI â€” FULL PIPELINE")
    print("=" * 70)

    # â”€â”€ Data acquisition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- STAGE 1/3: DATA ACQUISITION ---")
    math_path = stage_math(target_gb=1.0)
    nlp_path  = stage_nlp(target_gb=1.0)
    code_path = stage_code(target_gb=1.0)

    print()
    print("--- DATA SUMMARY ---")
    for name, path in [("math", math_path), ("nlp", nlp_path), ("code", code_path)]:
        if path.exists():
            gb   = path.stat().st_size / 1e9
            n    = sum(1 for _ in open(path, encoding="utf-8") if _.strip())
            print(f"  {name:<8} {gb:.3f} GB  {n:>9,} records  â†’ {path}")

    # â”€â”€ Pre-training analysis (untrained baseline) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- STAGE 2/3: PRE-TRAINING ANALYSIS (baseline) ---")
    models_to_train = {
        "math_v1":     (CellAI(params),   math_path, "problem",  "math"),
        "nlp_v1":      (CellAI(params),   nlp_path,  "text",     "nlp"),
        "code_v2":     (CellAIv2(params), code_path, "content",  "software"),
    }

    analysis_sample = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    analyses_before: Dict = {}
    for mname, (m, _, _, _) in models_to_train.items():
        analyses_before[mname] = analyse_model(m, analysis_sample, f"{mname}/before")

    # â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- STAGE 3/3: TRAINING ---")
    training_results: Dict = {}
    trained_models: Dict  = {}

    for mname, (model, data_path, tf, domain) in models_to_train.items():
        print(f"\n  Training {mname} ...")
        trainer = Trainer(
            model       = model,
            data_path   = data_path,
            text_field  = tf,
            lr          = 5e-4,
            max_steps   = 2000,          # ~10 min per model on RTX 3090
            segment_len = 64,
            log_every   = 200,
            ckpt_dir    = CKPT_DIR,
            name        = mname,
        )
        result = trainer.train()
        training_results[mname] = result
        trained_models[mname]   = (model, domain)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # â”€â”€ Post-training analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- POST-TRAINING ANALYSIS ---")
    analyses_after: Dict = {}
    for mname, (model, domain) in trained_models.items():
        analyses_after[mname] = analyse_model(model, analysis_sample, f"{mname}/after")

    # â”€â”€ Gradient norm comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- GRADIENT NORM CHANGES (before vs. after training) ---")
    for mname in models_to_train:
        before = analyses_before[mname]["grad_norms"]
        after  = analyses_after[mname]["grad_norms"]
        print(f"\n  {mname}:")
        for pname in sorted(before):
            b, a = before[pname], after.get(pname, 0)
            if b > 0 or a > 0:
                print(f"    {pname:<50} {b:.3e} â†’ {a:.3e}")

    # â”€â”€ Perplexity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- PERPLEXITY (lower = better; PPL capped at exp(30)~10^13 to avoid overflow) ---")
    ppl_results = {}
    for mname, (model, domain) in trained_models.items():
        data_path_map = {"math_v1": math_path, "nlp_v1": nlp_path, "code_v2": code_path}
        tf_map  = {"math_v1": "problem", "nlp_v1": "text", "code_v2": "content"}
        avg_loss, ppl = compute_perplexity(model, data_path_map[mname], tf_map[mname], n_samples=200)
        ppl_results[mname] = ppl
        print(f"  {mname:<16} avg_nll={avg_loss:.3f} nats/tok   PPL={ppl:,.0f}")

    # â”€â”€ Chat evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- CHAT EVALUATION ---")
    for mname, (model, domain) in trained_models.items():
        prompts = EVAL_PROMPTS.get(domain, EVAL_PROMPTS["nlp"])
        eval_chat(model, domain, prompts, mname)

    # â”€â”€ Multimodal model test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- MULTIMODAL MODEL TEST ---")
    mm_results = test_multimodal(params)

    # â”€â”€ Final summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  GPU:   {DEVICE}")
    print(f"  Data:")
    for name, path in [("math", math_path), ("nlp", nlp_path), ("code", code_path)]:
        print(f"    {name}: {path.stat().st_size/1e9:.3f} GB")
    print(f"  Training:")
    for mname, result in training_results.items():
        losses = result["losses"]
        first, last = (losses[0], losses[-1]) if losses else (float("nan"), float("nan"))
        print(f"    {mname:<16} loss: {first:.4f} â†’ {last:.4f}  "
              f"({'improved' if last < first else 'no improvement'})")
    print(f"  Perplexity (avg nll / PPL):")
    for mname, ppl in ppl_results.items():
        print(f"    {mname:<16} PPL={ppl:,.0f}")
    print(f"  Throughput (steady-state, v1):")
    for mname, a in analyses_after.items():
        print(f"    {mname:<16} {a['ms_per_forward']:.2f} ms/fwd")
    print("\nPIPELINE COMPLETE.")


if __name__ == "__main__":
    main()

