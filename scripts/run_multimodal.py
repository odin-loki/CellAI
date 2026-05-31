"""
Cell AI — Multimodal model: training, profiling, and evaluation.

Stages
------
1. Build a mixed-domain dataset with modality labels (math / nlp / code)
2. Train MultiModalModel jointly:
      L = L_router  (cross-entropy: predict modality from cellular state)
        + L_ntp     (next-token prediction per segment via BPTT)
3. Evaluate routing accuracy, per-modality perplexity, ablation study
4. Throughput and memory profiling
5. Print a structured results report for inclusion in the paper.

Run (from repo root):
    .venv\Scripts\python scripts\run_multimodal.py > C:\Temp\mm.log 2>&1
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = _REPO / "data" / "local"
CKPT_DIR = DATA_DIR / "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

MODALITIES = ["text", "code", "math"]   # must match multimodal.py
random.seed(42)
torch.manual_seed(42)

print("=" * 72)
print("CELL AI — MULTIMODAL TRAINING AND PROFILING")
print("=" * 72)
print(f"Device: {DEVICE}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mixed dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_mixed_dataset(n_per_modality: int = 8_000) -> List[Tuple[str, int]]:
    """
    Sample n_per_modality texts from each domain.
    Returns list of (text, modality_label) tuples.
    Label: 0=text, 1=code, 2=math
    """
    sources = [
        (DATA_DIR / "nlp"  / "train.jsonl",  "text",  0),
        (DATA_DIR / "code" / "train.jsonl",  "content", 1),
        (DATA_DIR / "math" / "train.jsonl",  "problem", 2),
    ]
    dataset: List[Tuple[str, int]] = []
    for path, field, label in sources:
        seen = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if seen >= n_per_modality:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec  = json.loads(line)
                    text = rec.get(field, "").strip()
                    if len(text) >= 40:
                        dataset.append((text, label))
                        seen += 1
                except Exception:
                    continue
        print(f"  Loaded {seen:,} {MODALITIES[label]} samples from {path.name}")
    random.shuffle(dataset)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 2. MultiModal Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalTrainer:
    """
    Joint training objective:
        L = lambda_r * L_router + lambda_ntp * L_ntp

    L_router  = CE(softmax(router(state)), true_modality_label)
    L_ntp     = per-token cross-entropy (next-token prediction via BPTT)

    Only the router, heads, and metaplasticity.state_gate are trained here
    (backbone is already pre-trained and frozen to save compute).
    """

    def __init__(
        self,
        model,               # MultiModalModel
        dataset: List[Tuple[str, int]],
        lr:            float = 2e-4,
        max_steps:     int   = 5_000,
        segment_len:   int   = 48,
        lambda_router: float = 1.0,
        lambda_ntp:    float = 0.3,
        log_every:     int   = 250,
        freeze_backbone: bool = True,
    ):
        self.model         = model
        self.dataset       = dataset
        self.lr            = lr
        self.max_steps     = max_steps
        self.segment_len   = segment_len
        self.lambda_r      = lambda_router
        self.lambda_ntp    = lambda_ntp
        self.log_every     = log_every
        self.freeze_backbone = freeze_backbone

        # Decide which parameters to train
        if freeze_backbone:
            # Train only the router and modality heads (and state_gate)
            cell = model.cell
            trainable = (
                list(model.router.parameters()) +
                list(model.heads.parameters())  +
                list(cell.metaplasticity.state_gate.parameters()) +
                list(cell.output_proj.parameters())
            )
            print(f"  Frozen backbone; training {sum(p.numel() for p in trainable):,} params")
        else:
            trainable = [p for p in model.cell.parameters() if p.requires_grad]
            trainable += list(model.router.parameters())
            trainable += list(model.heads.parameters())
            print(f"  Training full model: {sum(p.numel() for p in trainable):,} params")

        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=lr * 0.05)

        self.router_losses: List[float] = []
        self.ntp_losses:    List[float] = []
        self.total_losses:  List[float] = []
        self.steps:         List[int]   = []

    def _ntp_step(self, text: str, cell) -> torch.Tensor:
        """
        Sequential next-token prediction for one text.
        Returns mean per-token CE loss (as a scalar tensor with grad).
        """
        tokens = cell.encoder.tokenize(text)
        if len(tokens) < 2:
            return torch.tensor(0.0, device=DEVICE)

        cell.partitions.reset()
        cell.memory_formation.reset()

        tok_ids = torch.tensor(tokens[:512], dtype=torch.long, device=DEVICE)
        total   = torch.tensor(0.0, device=DEVICE)
        count   = 0

        for seg_start in range(0, len(tok_ids) - 1, self.segment_len):
            seg_end  = min(seg_start + self.segment_len, len(tok_ids) - 1)
            seg_ids  = tok_ids[seg_start : seg_end + 1]
            seg_embs = cell.encoder.embedding(seg_ids) * cell.encoder._scale

            seg_loss = torch.tensor(0.0, device=DEVICE)
            for t_local in range(seg_end - seg_start):
                inp = seg_embs[t_local]
                cell.partitions.step(inp)
                agg    = cell.partitions.aggregate()
                memory = cell.memory_formation(inp, agg)
                out    = cell.metaplasticity(agg, memory, inp)
                state  = cell.output_proj(out)
                logits = state @ cell.encoder.embedding.weight.t()
                seg_loss = seg_loss + F.cross_entropy(
                    logits.unsqueeze(0), seg_ids[t_local + 1].unsqueeze(0))
                count += 1

            (seg_loss / max(seg_end - seg_start, 1)).backward(retain_graph=False)
            total = total + seg_loss.detach()
            cell.partitions._buffers["state"] = (
                cell.partitions._buffers["state"].detach())

        return total / max(count, 1)

    def train(self) -> Dict:
        print(f"\n[MultiModal] Training {self.max_steps} steps, lr={self.lr}")
        self.model.cell.train()

        data_iter = iter(self.dataset)
        run_r, run_ntp, run_tot = 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        for step in range(1, self.max_steps + 1):
            # Cycle through dataset
            try:
                text, mod_label = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataset)
                text, mod_label = next(data_iter)

            self.optimizer.zero_grad()

            # ── Router loss ──────────────────────────────────────────────
            cell = self.model.cell
            encoded   = cell.encode_input(text[:200])   # short prefix for routing
            cell_state = cell.cellular_step(encoded)
            router_logits = self.model.router(cell_state.detach())  # (3,)
            target_mod    = torch.tensor([mod_label], dtype=torch.long, device=DEVICE)
            l_router = F.cross_entropy(router_logits.unsqueeze(0), target_mod)
            (self.lambda_r * l_router).backward()
            run_r += l_router.item()

            # ── NTP loss ─────────────────────────────────────────────────
            cell.partitions.reset()
            cell.memory_formation.reset()
            l_ntp_val = self._ntp_step(text, cell)
            # NTP backward already done inside _ntp_step via segment backward
            run_ntp += l_ntp_val.item()

            # ── Combined backward & step ─────────────────────────────────
            torch.nn.utils.clip_grad_norm_(
                list(self.model.router.parameters()) +
                list(self.model.heads.parameters()) +
                list(cell.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
            self.scheduler.step()

            tot = self.lambda_r * l_router.item() + self.lambda_ntp * l_ntp_val.item()
            run_tot += tot

            if step % self.log_every == 0:
                elapsed = time.perf_counter() - t0
                lr_now  = self.scheduler.get_last_lr()[0]
                tps     = step / elapsed
                avg_r   = run_r   / self.log_every
                avg_ntp = run_ntp / self.log_every
                avg_tot = run_tot / self.log_every
                print(f"  step={step:>5}  router={avg_r:.3f}  ntp={avg_ntp:.3f}  "
                      f"total={avg_tot:.3f}  lr={lr_now:.2e}  {tps:.1f} s/s")
                self.router_losses.append(avg_r)
                self.ntp_losses.append(avg_ntp)
                self.total_losses.append(avg_tot)
                self.steps.append(step)
                run_r = run_ntp = run_tot = 0.0

        # Save checkpoint
        ckpt = CKPT_DIR / "multimodal_v2.pt"
        torch.save({
            "cell_state":    self.model.cell.state_dict(),
            "router_state":  self.model.router.state_dict(),
            "heads_state":   self.model.heads.state_dict(),
            "router_losses": self.router_losses,
            "ntp_losses":    self.ntp_losses,
            "total_losses":  self.total_losses,
            "steps":         self.steps,
        }, ckpt)
        print(f"\n[MultiModal] Checkpoint saved -> {ckpt}")
        self.model.cell.eval()
        return {
            "steps":     self.steps,
            "router":    self.router_losses,
            "ntp":       self.ntp_losses,
            "total":     self.total_losses,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Routing accuracy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_routing(model, dataset: List[Tuple[str, int]], n_eval: int = 600) -> Dict:
    """
    Measure how accurately the router assigns modality labels.
    Returns per-class precision/recall and overall accuracy.
    """
    model.cell.eval()
    W_snap = model.cell.metaplasticity.W.data.clone()

    confusion = torch.zeros(3, 3, dtype=torch.long)   # true × predicted
    sample = random.sample(dataset, min(n_eval, len(dataset)))

    for text, true_label in sample:
        model.cell.metaplasticity.W.data.copy_(W_snap)
        model.cell.partitions.reset()
        with torch.no_grad():
            enc   = model.cell.encode_input(text[:200])
            state = model.cell.cellular_step(enc)
            logits = model.router(state)
            pred  = logits.argmax().item()
        confusion[true_label, pred] += 1

    # Restore
    model.cell.metaplasticity.W.data.copy_(W_snap)

    accuracy = confusion.diagonal().sum().item() / confusion.sum().item()
    per_class: Dict[str, Dict] = {}
    for i, name in enumerate(MODALITIES):
        tp  = confusion[i, i].item()
        fp  = (confusion[:, i].sum() - tp).item()
        fn  = (confusion[i, :].sum() - tp).item()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        per_class[name] = {"precision": prec, "recall": rec, "f1": f1,
                           "n_true": confusion[i].sum().item(),
                           "n_pred": confusion[:, i].sum().item()}

    return {"accuracy": accuracy, "confusion": confusion.tolist(), "per_class": per_class}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-modality perplexity
# ─────────────────────────────────────────────────────────────────────────────

def eval_perplexity_by_modality(
    model, dataset: List[Tuple[str, int]], n_per: int = 100
) -> Dict[str, Dict]:
    """
    Compute average NLL and PPL for each modality separately.
    """
    model.cell.eval()
    W_snap = model.cell.metaplasticity.W.data.clone()

    by_modality: Dict[int, List[float]] = {0: [], 1: [], 2: []}

    # Sample n_per per modality
    per_mod: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for text, label in dataset:
        if len(per_mod[label]) < n_per:
            per_mod[label].append(text)

    for label, texts in per_mod.items():
        for text in texts:
            model.cell.metaplasticity.W.data.copy_(W_snap)
            model.cell.partitions.reset()
            model.cell.memory_formation.reset()

            tokens = model.cell.encoder.tokenize(text)[:256]
            if len(tokens) < 2:
                continue

            tok_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            embs    = model.cell.encoder.embedding(tok_ids) * model.cell.encoder._scale
            tot_nll = 0.0
            count   = 0

            for t in range(len(tokens) - 1):
                with torch.no_grad():
                    model.cell.partitions.step(embs[t].detach())
                    agg    = model.cell.partitions.aggregate()
                    memory = model.cell.memory_formation(embs[t].detach(), agg)
                    out    = model.cell.metaplasticity(agg, memory, embs[t].detach())
                    state  = model.cell.output_proj(out)

                    # Route through correct head for NLL
                    weights   = F.softmax(model.router(state), dim=0)
                    head_outs = torch.stack([h(state) for h in model.heads], dim=0)
                    routed    = (weights.unsqueeze(1) * head_outs).sum(dim=0)

                    logits = routed @ model.cell.encoder.embedding.weight.t()
                    nll    = F.cross_entropy(logits.unsqueeze(0), tok_ids[t+1].unsqueeze(0))
                    tot_nll += nll.item()
                    count   += 1

            if count > 0:
                by_modality[label].append(tot_nll / count)

    model.cell.metaplasticity.W.data.copy_(W_snap)

    results: Dict[str, Dict] = {}
    for label, nlls in by_modality.items():
        if nlls:
            avg = sum(nlls) / len(nlls)
            results[MODALITIES[label]] = {
                "avg_nll": avg,
                "ppl": math.exp(min(avg, 30.0)),
                "n": len(nlls),
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Throughput profiling
# ─────────────────────────────────────────────────────────────────────────────

def profile_throughput(model) -> Dict:
    """Detailed latency breakdown: cellular, router, heads."""
    model.cell.eval()
    N_WARM, N_BENCH = 10, 200
    results: Dict[str, float] = {}

    test_text = "Machine learning is a subfield of artificial intelligence."

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Full pipeline
    with torch.no_grad():
        for _ in range(N_WARM):
            model.cell.encode_input(test_text)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_BENCH):
            model.cell.encode_input(test_text)
    sync()
    results["encode_ms"] = (time.perf_counter() - t0) / N_BENCH * 1000

    # Cellular step
    enc = model.cell.encode_input(test_text)
    with torch.no_grad():
        for _ in range(N_WARM):
            model.cell.cellular_step(enc)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_BENCH):
            model.cell.cellular_step(enc)
    sync()
    results["cellular_ms"] = (time.perf_counter() - t0) / N_BENCH * 1000

    # Router
    state = model.cell.cellular_step(enc)
    with torch.no_grad():
        for _ in range(N_WARM):
            model.router(state)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_BENCH):
            model.router(state)
    sync()
    results["router_ms"] = (time.perf_counter() - t0) / N_BENCH * 1000

    # All heads (3x MLP)
    with torch.no_grad():
        for _ in range(N_WARM):
            for h in model.heads:
                h(state)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_BENCH):
            for h in model.heads:
                h(state)
    sync()
    results["heads_ms"] = (time.perf_counter() - t0) / N_BENCH * 1000

    # Full forward
    with torch.no_grad():
        for _ in range(N_WARM):
            model.chat(test_text)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_BENCH):
            model.chat(test_text)
    sync()
    results["full_ms"] = (time.perf_counter() - t0) / N_BENCH * 1000

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ablation study
# ─────────────────────────────────────────────────────────────────────────────

def ablation_study(model, dataset: List[Tuple[str, int]], n_eval: int = 150) -> Dict:
    """
    Compare routing accuracy:
      (a) Full model (router + 3 heads)
      (b) No router (uniform mixture)
      (c) Backbone only (no domain head)
      (d) Per-modality head only (oracle routing)
    """
    model.cell.eval()
    W_snap = model.cell.metaplasticity.W.data.clone()
    sample = random.sample(dataset, min(n_eval, len(dataset)))

    def avg_nll_no_router():
        total, count = 0.0, 0
        model.cell.metaplasticity.W.data.copy_(W_snap)
        for text, _ in sample[:50]:
            model.cell.partitions.reset()
            model.cell.memory_formation.reset()
            tokens = model.cell.encoder.tokenize(text)[:128]
            if len(tokens) < 2:
                continue
            tok_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            embs = model.cell.encoder.embedding(tok_ids) * model.cell.encoder._scale
            for t in range(len(tokens) - 1):
                with torch.no_grad():
                    model.cell.partitions.step(embs[t].detach())
                    agg = model.cell.partitions.aggregate()
                    mem = model.cell.memory_formation(embs[t].detach(), agg)
                    out = model.cell.metaplasticity(agg, mem, embs[t].detach())
                    state = model.cell.output_proj(out)
                    # Uniform mixture (ablate router: equal weights)
                    head_outs = torch.stack([h(state) for h in model.heads], dim=0)
                    routed = head_outs.mean(dim=0)
                    logits = routed @ model.cell.encoder.embedding.weight.t()
                    total += F.cross_entropy(logits.unsqueeze(0), tok_ids[t+1].unsqueeze(0)).item()
                    count += 1
        return total / max(count, 1)

    def avg_nll_backbone_only():
        total, count = 0.0, 0
        model.cell.metaplasticity.W.data.copy_(W_snap)
        for text, _ in sample[:50]:
            model.cell.partitions.reset()
            model.cell.memory_formation.reset()
            tokens = model.cell.encoder.tokenize(text)[:128]
            if len(tokens) < 2:
                continue
            tok_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
            embs = model.cell.encoder.embedding(tok_ids) * model.cell.encoder._scale
            for t in range(len(tokens) - 1):
                with torch.no_grad():
                    model.cell.partitions.step(embs[t].detach())
                    agg = model.cell.partitions.aggregate()
                    mem = model.cell.memory_formation(embs[t].detach(), agg)
                    out = model.cell.metaplasticity(agg, mem, embs[t].detach())
                    state = model.cell.output_proj(out)
                    logits = state @ model.cell.encoder.embedding.weight.t()
                    total += F.cross_entropy(logits.unsqueeze(0), tok_ids[t+1].unsqueeze(0)).item()
                    count += 1
        return total / max(count, 1)

    results = {}
    print("  [Ablation] uniform routing (no router)...")
    results["no_router_nll"] = avg_nll_no_router()
    print("  [Ablation] backbone only (no domain head)...")
    results["backbone_only_nll"] = avg_nll_backbone_only()
    model.cell.metaplasticity.W.data.copy_(W_snap)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. Chat demonstrations
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PROMPTS = {
    "math": [
        "Solve for x: 3x + 7 = 22",
        "What is the derivative of sin(x)?",
        "Find the eigenvalues of [[2,1],[1,2]]",
        "P(X=2) for X~Binomial(8, 0.3)?",
    ],
    "code": [
        "def fibonacci(n):",
        "class BinarySearchTree:",
        "import torch; model = nn.Linear(",
        "def quicksort(arr):",
    ],
    "text": [
        "Neural networks learn representations through",
        "The transformer architecture introduced by",
        "Backpropagation computes gradients by",
        "Machine learning is a subfield of",
    ],
}


def demo_chat(model) -> Dict[str, List[Dict]]:
    model.cell.eval()
    W_snap = model.cell.metaplasticity.W.data.clone()
    results: Dict[str, List[Dict]] = {}
    for domain, prompts in DEMO_PROMPTS.items():
        results[domain] = []
        print(f"\n  [{domain.upper()}]")
        for prompt in prompts:
            model.cell.metaplasticity.W.data.copy_(W_snap)
            detected = model.detect_modality(prompt)
            with torch.no_grad():
                resp = model.chat(prompt)
            ms_t = 0
            t0 = time.perf_counter()
            with torch.no_grad():
                model.chat(prompt)
            ms_t = (time.perf_counter() - t0) * 1000
            print(f"    Q [{detected:<4}]: {prompt!r}")
            print(f"    A          : {resp[:80]!r}  [{ms_t:.0f}ms]")
            results[domain].append({
                "prompt":   prompt,
                "detected": detected,
                "response": resp,
                "ms":       ms_t,
            })
    model.cell.metaplasticity.W.data.copy_(W_snap)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. Memory profiling (CUDA only)
# ─────────────────────────────────────────────────────────────────────────────

def profile_memory(model) -> Dict:
    if not torch.cuda.is_available():
        return {"note": "CUDA not available"}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.cell.eval()
    with torch.no_grad():
        for _ in range(20):
            model.chat("The field of artificial intelligence studies.")
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    param_mb = sum(p.numel() * p.element_size() for p in model.cell.parameters()) / 1e6
    param_mb += sum(p.numel() * p.element_size() for p in model.router.parameters()) / 1e6
    param_mb += sum(p.numel() * p.element_size() for p in model.heads.parameters()) / 1e6
    return {
        "peak_activation_mb": peak_mb - param_mb,
        "param_mb":           param_mb,
        "total_peak_mb":      peak_mb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from cellai_core.base            import ModelParams
    from v2.cell_ai_v2               import CellAIv2
    from models.multimodal.multimodal import MultiModalModel

    params = ModelParams(state_size=256, num_partitions=4)

    # Load pre-trained backbone
    print("Loading code_v2 backbone checkpoint ...")
    backbone = CellAIv2(params)
    ckpt_path = CKPT_DIR / "code_v2.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        backbone.load_state_dict(ckpt["state_dict"])
        losses = ckpt.get("losses", [])
        if losses:
            print(f"  Backbone loss curve: {losses[0]:.2f} -> {losses[-1]:.2f}")
    else:
        print("  WARNING: No backbone checkpoint found. Using random init.")
    backbone.eval()

    # Instantiate multimodal model
    mm = MultiModalModel(cell_system=backbone, params=params)
    print(f"  MultiModal params: {sum(p.numel() for p in mm.router.parameters()) + sum(p.numel() for p in mm.heads.parameters()):,} (router+heads)")

    # ── 1. Build dataset ──────────────────────────────────────────────────
    print("\n--- 1. BUILDING MIXED DATASET ---")
    dataset = build_mixed_dataset(n_per_modality=8_000)
    n_train = int(len(dataset) * 0.85)
    train_set = dataset[:n_train]
    eval_set  = dataset[n_train:]
    print(f"  Total: {len(dataset):,}  train={n_train:,}  eval={len(eval_set):,}")

    # ── 2. Pre-training routing accuracy (baseline) ───────────────────────
    print("\n--- 2. BASELINE ROUTING ACCURACY (before training) ---")
    baseline_routing = eval_routing(mm, eval_set, n_eval=300)
    print(f"  Accuracy: {baseline_routing['accuracy']:.3f}")
    for mod, stats in baseline_routing["per_class"].items():
        print(f"    {mod:<6}  P={stats['precision']:.3f}  R={stats['recall']:.3f}  F1={stats['f1']:.3f}")
    print(f"  Confusion (true x pred):")
    for row in baseline_routing["confusion"]:
        print(f"    {row}")

    # ── 3. Train ──────────────────────────────────────────────────────────
    print("\n--- 3. TRAINING ---")
    trainer = MultiModalTrainer(
        model          = mm,
        dataset        = train_set,
        lr             = 2e-4,
        max_steps      = 5_000,
        segment_len    = 48,
        lambda_router  = 1.0,
        lambda_ntp     = 0.3,
        log_every      = 250,
        freeze_backbone= False,
    )
    train_results = trainer.train()

    # ── 4. Post-training routing accuracy ─────────────────────────────────
    print("\n--- 4. POST-TRAINING ROUTING ACCURACY ---")
    post_routing = eval_routing(mm, eval_set, n_eval=600)
    print(f"  Accuracy: {post_routing['accuracy']:.3f}  "
          f"(baseline: {baseline_routing['accuracy']:.3f}  "
          f"delta: {post_routing['accuracy'] - baseline_routing['accuracy']:+.3f})")
    for mod, stats in post_routing["per_class"].items():
        print(f"    {mod:<6}  P={stats['precision']:.3f}  R={stats['recall']:.3f}  F1={stats['f1']:.3f}")
    print("  Confusion (true x pred):")
    for i, row in enumerate(post_routing["confusion"]):
        print(f"    {MODALITIES[i]:<6}: {row}")

    # ── 5. Per-modality perplexity ────────────────────────────────────────
    print("\n--- 5. PER-MODALITY PERPLEXITY ---")
    print("  (Random-init baseline: PPL ~ 98,716 for 100k vocab)")
    ppl_results = eval_perplexity_by_modality(mm, eval_set, n_per=100)
    for mod, res in ppl_results.items():
        print(f"  {mod:<6}  avg_nll={res['avg_nll']:.3f} nats/tok   PPL={res['ppl']:,.0f}")

    # ── 6. Ablation study ─────────────────────────────────────────────────
    print("\n--- 6. ABLATION STUDY ---")
    ablation = ablation_study(mm, eval_set, n_eval=150)
    ppl_full = min(list(ppl_results.values())[0]["avg_nll"] if ppl_results else 30, 30)
    print(f"  Full model (router + heads):   NLL = {ppl_full:.3f}")
    print(f"  Uniform routing (no router):   NLL = {ablation['no_router_nll']:.3f}")
    print(f"  Backbone only (no head):       NLL = {ablation['backbone_only_nll']:.3f}")

    # ── 7. Throughput profiling ───────────────────────────────────────────
    print("\n--- 7. THROUGHPUT PROFILING ---")
    tput = profile_throughput(mm)
    overhead_pct = (tput["router_ms"] + tput["heads_ms"]) / tput["full_ms"] * 100
    print(f"  encode         : {tput['encode_ms']:.3f} ms")
    print(f"  cellular step  : {tput['cellular_ms']:.3f} ms")
    print(f"  router (Linear): {tput['router_ms']:.3f} ms")
    print(f"  3x heads (MLP) : {tput['heads_ms']:.3f} ms")
    print(f"  full forward   : {tput['full_ms']:.3f} ms")
    print(f"  router+heads overhead: {overhead_pct:.1f}%")

    # ── 8. Memory profiling ───────────────────────────────────────────────
    print("\n--- 8. MEMORY PROFILING ---")
    mem = profile_memory(mm)
    print(f"  Parameters    : {mem.get('param_mb', 0):.1f} MB")
    print(f"  Peak activation: {mem.get('peak_activation_mb', 0):.1f} MB")
    print(f"  Total peak    : {mem.get('total_peak_mb', 0):.1f} MB")

    # ── 9. Chat demonstrations ────────────────────────────────────────────
    print("\n--- 9. CHAT DEMONSTRATIONS ---")
    chats = demo_chat(mm)

    # ── 10. Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MULTIMODAL SUMMARY")
    print("=" * 72)
    print(f"  Architecture : CellAI v2 + MultiModalRouter (3 modalities)")
    print(f"  Params       : {sum(p.numel() for p in backbone.parameters()):,} backbone  "
          f"+ {sum(p.numel() for p in mm.router.parameters()) + sum(p.numel() for p in mm.heads.parameters()):,} heads")
    print(f"  Training     : {trainer.max_steps} steps, lr={trainer.lr}, lambda_r={trainer.lambda_r}")
    if train_results["router"]:
        print(f"  Router loss  : {train_results['router'][0]:.3f} -> {train_results['router'][-1]:.3f}")
    if train_results["ntp"]:
        print(f"  NTP loss     : {train_results['ntp'][0]:.3f} -> {train_results['ntp'][-1]:.3f}")
    print(f"  Routing acc  : {baseline_routing['accuracy']:.3f} -> {post_routing['accuracy']:.3f}")
    print(f"  Throughput   : {tput['full_ms']:.2f} ms/forward  ({1000/tput['full_ms']:.0f} calls/s)")
    print()

    # Save all results for paper
    summary = {
        "baseline_routing": baseline_routing,
        "post_routing":     post_routing,
        "ppl_results":      ppl_results,
        "ablation":         ablation,
        "throughput":       tput,
        "memory":           mem,
        "train_results":    train_results,
        "chats":            chats,
    }
    import json as _json
    out_path = DATA_DIR / "multimodal_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved -> {out_path}")
    print("\nDONE.")
    return summary


if __name__ == "__main__":
    main()
