r"""
Cell AI v3 — Guided Architecture Search
========================================
8 experiments testing different architectural improvements, each run for 2000 steps
on balanced multi-domain data, profiled end-to-end.

Experiments
-----------
E0  Fixed baseline    — multi-domain training, all bug fixes, dense PDE + full Hebbian
E1  SpectralPDE       — FFT-based O(D log D) diffusion, vs dense O(D²)
E2  SparseHebbian     — top-k (k=D/8) synaptic updates, O(D log D)
E3  MultiScale        — fast/slow partition hierarchy (4×128 + 2×256)
E4  EntropyRouter     — Gumbel-Softmax + load-balance loss (multimodal)
E5  PerFreqResonance  — per-frequency complex FFT gate (v2 resonance upgrade)
E6  LargeScale        — D=512, N=8 partitions
E7  Combined          — SpectralPDE + SparseHebbian + PerFreqResonance, D=512

For E4 (router), the model is the combined backbone + EntropyRouter + 3 heads.
All other experiments are backbone only (NTP training).

Results are written to data/local/arch_search/results.json at the end.

Run (from repo root):
    .venv\Scripts\python -m arch_search.run_arch_search > C:\Temp\arch.log 2>&1
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
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
OUT_DIR  = DATA_DIR / "arch_search"
os.makedirs(OUT_DIR, exist_ok=True)

MODALITY_FIELDS = [
    (DATA_DIR / "nlp"  / "train.jsonl",  "text",    0),
    (DATA_DIR / "code" / "train.jsonl",  "content", 1),
    (DATA_DIR / "math" / "train.jsonl",  "problem", 2),
]
MODALITY_NAMES = ["text", "code", "math"]

random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True   # maximise GPU throughput

print("=" * 72)
print("CELL AI v3 — GUIDED ARCHITECTURE SEARCH")
print("=" * 72)
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Shared data loader (balanced multi-domain)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(n_per: int = 6_000) -> List[Tuple[str, int]]:
    """Load n_per texts per domain, shuffle, return [(text, label), ...]."""
    dataset: List[Tuple[str, int]] = []
    for path, field, label in MODALITY_FIELDS:
        seen = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if seen >= n_per:
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
        print(f"  Loaded {seen:,} {MODALITY_NAMES[label]} samples")
    random.shuffle(dataset)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 2. Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Trains a CellAIv3 model on balanced multi-domain data and collects metrics.
    """

    TRAIN_STEPS = 2_000
    SEGMENT_LEN = 64
    LR          = 3e-4
    LOG_EVERY   = 200

    def __init__(self, dataset: List[Tuple[str, int]]):
        n_train = int(len(dataset) * 0.85)
        self.train_set = dataset[:n_train]
        self.eval_set  = dataset[n_train:]

    # ------------------------------------------------------------------

    def train(self, model: nn.Module) -> Dict:
        """Train for TRAIN_STEPS steps and return loss history."""
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr = self.LR, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.TRAIN_STEPS, eta_min=self.LR * 0.05
        )

        # Shuffle once before training to interleave all three domains.
        # Without shuffling, sequential domain blocks cause large oscillations in
        # the 200-step averages (math NLL≈4, code NLL≈7), making convergence hard
        # to diagnose.  Shuffling is idempotent for the model but improves signal.
        train_shuffled = list(self.train_set)
        random.shuffle(train_shuffled)

        data_iter  = iter(train_shuffled)
        losses: List[float] = []
        steps:  List[int]   = []
        run_loss = 0.0
        t0 = time.perf_counter()

        for step in range(1, self.TRAIN_STEPS + 1):
            if step == 1:
                print(
                    f"    … step 1/{self.TRAIN_STEPS} started (next log at step {self.LOG_EVERY})",
                    flush=True,
                )
            try:
                text, _ = next(data_iter)
            except StopIteration:
                random.shuffle(train_shuffled)
                data_iter = iter(train_shuffled)
                text, _ = next(data_iter)

            loss = model.train_step_sequential(
                text, optimizer, segment_len=self.SEGMENT_LEN, reset_state=True
            )
            scheduler.step()
            run_loss += loss

            if step % self.LOG_EVERY == 0:
                avg = run_loss / self.LOG_EVERY
                lr  = scheduler.get_last_lr()[0]
                tps = step / (time.perf_counter() - t0)
                print(f"    step={step:>5}  loss={avg:.3f}  lr={lr:.2e}  {tps:.1f} s/s", flush=True)
                losses.append(avg)
                steps.append(step)
                run_loss = 0.0

        return {"steps": steps, "losses": losses,
                "initial": losses[0] if losses else float("nan"),
                "final":   losses[-1] if losses else float("nan"),
                "pct_reduction": (1 - losses[-1]/losses[0]) * 100 if losses else 0}

    # ------------------------------------------------------------------

    def eval_ppl(self, model: nn.Module, n: int = 150) -> Dict:
        """Per-modality perplexity on held-out data."""
        model.eval()
        W_snap = model.metaplasticity.W.data.clone()

        per_mod: Dict[int, List[float]] = {0: [], 1: [], 2: []}
        samples = {label: [t for t, l in self.eval_set if l == label][:n//3]
                   for label in range(3)}

        for label, texts in samples.items():
            for text in texts:
                model.metaplasticity.W.data.copy_(W_snap)
                model.partitions.reset()
                model.memory_formation.reset()

                toks = model.encoder.tokenize(text)[:256]
                if len(toks) < 2:
                    continue
                tok_ids = torch.tensor(toks, dtype=torch.long, device=DEVICE)
                embs = model.encoder.embedding(tok_ids) * model.encoder._scale
                nll_total, count = 0.0, 0

                for t in range(len(toks) - 1):
                    with torch.no_grad():
                        # Use cellular_step so extensions (resonance etc.) are included
                        state  = model.cellular_step(embs[t])
                        logits = state @ model.encoder.embedding.weight.t()
                        nll    = F.cross_entropy(logits.unsqueeze(0), tok_ids[t+1].unsqueeze(0))
                        nll_total += nll.item()
                        count += 1

                if count > 0:
                    per_mod[label].append(nll_total / count)

        model.metaplasticity.W.data.copy_(W_snap)

        results = {}
        for label, nlls in per_mod.items():
            if nlls:
                avg = sum(nlls) / len(nlls)
                results[MODALITY_NAMES[label]] = {
                    "avg_nll": avg,
                    "ppl": math.exp(min(avg, 30.0)),
                    "n": len(nlls),
                }
        # Macro-average
        all_nlls = [v["avg_nll"] for v in results.values()]
        results["macro_avg_nll"] = sum(all_nlls) / len(all_nlls) if all_nlls else float("nan")
        results["macro_ppl"]     = math.exp(min(results["macro_avg_nll"], 30.0))
        return results

    # ------------------------------------------------------------------

    def profile_throughput(self, model: nn.Module, n: int = 300) -> Dict:
        """Measure latency of each component separately."""
        model.eval()
        text = "The reaction diffusion equation governs pattern formation."

        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(20):
                model.forward(text)
        sync()

        # Encode
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                model.encoder.encode_pooled(text, device=DEVICE)
        sync()
        encode_ms = (time.perf_counter() - t0) / n * 1000

        # Cellular step (encode once)
        enc = model.encoder.encode_pooled(text, device=DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                model.partitions.step(enc)
                model.partitions.aggregate()
        sync()
        partition_ms = (time.perf_counter() - t0) / n * 1000

        # Full forward
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                model.forward(text)
        sync()
        full_ms = (time.perf_counter() - t0) / n * 1000

        return {
            "encode_ms":    encode_ms,
            "partition_ms": partition_ms,
            "full_ms":      full_ms,
            "calls_per_s":  1000 / full_ms,
        }

    # ------------------------------------------------------------------

    def grad_analysis(self, model: nn.Module) -> Dict:
        """Gradient norm analysis after one training step."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        text, _ = self.train_set[0]
        model.train_step_sequential(text, optimizer, segment_len=64, reset_state=True)

        grad_norms = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norms[name] = p.grad.norm().item()
            else:
                grad_norms[name] = 0.0

        # Aggregate by module
        agg = {}
        for name, norm in grad_norms.items():
            module = name.split(".")[0]
            if module not in agg:
                agg[module] = []
            agg[module].append(norm)

        summary = {k: sum(v)/len(v) for k, v in agg.items()}
        summary["min"] = min(grad_norms.values()) if grad_norms else 0.0
        summary["max"] = max(grad_norms.values()) if grad_norms else 0.0
        summary["dead_params"] = sum(1 for v in grad_norms.values() if v < 1e-9)
        summary["total_params"] = len(grad_norms)
        return summary

    # ------------------------------------------------------------------

    def eval_generation(self, model: nn.Module, n_prompts: int = 6) -> List[Dict]:
        """Test autoregressive generation quality."""
        prompts = [
            ("math", "Solve for x: 2x + 5 = 13"),
            ("math", "The derivative of x squared is"),
            ("code", "def fibonacci(n):\n    if n <= 1: return n\n    return"),
            ("code", "class Node:\n    def __init__(self, val):"),
            ("text", "The transformer architecture learns"),
            ("text", "Neural networks are trained by"),
        ]

        results = []
        W_snap = model.metaplasticity.W.data.clone()

        for domain, prompt in prompts[:n_prompts]:
            model.metaplasticity.W.data.copy_(W_snap)
            try:
                t0  = time.perf_counter()
                gen = model.generate(prompt, max_tokens=32, temperature=0.8, top_p=0.9)
                ms  = (time.perf_counter() - t0) * 1000
                continuation = gen[len(prompt):]
                results.append({
                    "domain":       domain,
                    "prompt":       prompt,
                    "continuation": continuation,
                    "ms":           ms,
                })
                print(f"    [{domain}] {prompt!r}")
                print(f"           → {continuation[:60]!r}  [{ms:.0f}ms]")
            except Exception as e:
                results.append({"domain": domain, "prompt": prompt, "error": str(e)})

        model.metaplasticity.W.data.copy_(W_snap)
        return results

    # ------------------------------------------------------------------

    def memory_stats(self, model: nn.Module) -> Dict:
        if not torch.cuda.is_available():
            return {"note": "CUDA not available"}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        model.eval()
        text = "benchmark"
        with torch.no_grad():
            for _ in range(10):
                model.forward(text)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        return {
            "param_mb":   param_mb,
            "peak_mb":    peak_mb,
            "n_params":   sum(p.numel() for p in model.parameters()),
        }

    # ------------------------------------------------------------------

    def run_experiment(self, model: nn.Module, exp_name: str, n_steps: int = 0) -> Dict:
        """Full experiment: train → ppl → throughput → grads → generate.
        n_steps=0 uses the class default TRAIN_STEPS (2000).
        """
        print(f"\n{'='*68}")
        print(f"  {exp_name}  —  {model.get_info()}")
        print(f"{'='*68}")

        info     = model.get_info()
        print(f"  Parameters: {info['n_params']:,}")

        print("\n  [Training]")
        if n_steps > 0:
            orig = self.TRAIN_STEPS
            self.TRAIN_STEPS = n_steps
            train_r = self.train(model)
            self.TRAIN_STEPS = orig
        else:
            train_r = self.train(model)

        print("\n  [Perplexity eval]")
        ppl_r    = self.eval_ppl(model)
        for mod, stats in ppl_r.items():
            if isinstance(stats, dict):
                print(f"    {mod:<8}  nll={stats['avg_nll']:.3f}  ppl={stats['ppl']:,.0f}")
        print(f"    Macro NLL = {ppl_r.get('macro_avg_nll', 'n/a'):.3f}")

        print("\n  [Throughput]")
        tput_r   = self.profile_throughput(model)
        print(f"    Partition step: {tput_r['partition_ms']:.3f} ms")
        print(f"    Full forward:   {tput_r['full_ms']:.3f} ms  ({tput_r['calls_per_s']:.0f} calls/s)")

        print("\n  [Gradient analysis]")
        grad_r = self.grad_analysis(model)
        print(f"    Dead params: {grad_r['dead_params']}/{grad_r['total_params']}")
        for k, v in grad_r.items():
            if k not in ("min", "max", "dead_params", "total_params") and isinstance(v, float):
                print(f"    {k}: {v:.4e}")

        print("\n  [Autoregressive generation]")
        gen_r = self.eval_generation(model)

        print("\n  [Memory]")
        mem_r = self.memory_stats(model)
        print(f"    Params: {mem_r.get('param_mb', 0):.1f} MB  Peak: {mem_r.get('peak_mb', 0):.1f} MB")

        result = {
            "name":       exp_name,
            "info":       info,
            "train":      train_r,
            "ppl":        ppl_r,
            "throughput": tput_r,
            "gradients":  grad_r,
            "generation": gen_r,
            "memory":     mem_r,
        }

        # Save checkpoint
        ckpt_path = OUT_DIR / f"{exp_name.replace(' ', '_')}.pt"
        torch.save({"state_dict": model.state_dict(), "result": result}, ckpt_path)
        print(f"\n  Checkpoint -> {ckpt_path}")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Experiment definitions
# ─────────────────────────────────────────────────────────────────────────────

def make_model(params, cfg):
    """Build CellAIv3 and move to DEVICE."""
    from v3.cell_ai_v3 import CellAIv3
    m = CellAIv3(params, cfg)
    m.to(DEVICE)
    return m


def run_e4_multimodal(runner: ExperimentRunner) -> Dict:
    """
    E4: Entropy-regularized router.
    Train the backbone on balanced data, then add the entropy router
    and fine-tune router + heads for 1000 additional steps.
    """
    from cellai_core.base    import ModelParams
    from cellai_core.routing import AnnealedRouter
    from v3.cell_ai_v3       import CellAIv3, V3Config

    print("\n" + "="*68)
    print("  E4 — EntropyRouter (Gumbel-Softmax + load-balance)")
    print("="*68)

    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E4_entropyrouter", pde_type="dense", hebbian_type="full")
    backbone = CellAIv3(params, cfg).to(DEVICE)

    # Step 1: train backbone on balanced data for 2000 steps
    print("\n  [Backbone training]")
    bb_result = runner.train(backbone)

    # Step 2: attach router + 3 domain heads
    D = 256
    N_MOD = 3
    router = AnnealedRouter(D, N_MOD, T_start=2.0, T_end=0.5,
                            anneal_steps=1000, lambda_balance=0.02).to(DEVICE)

    class ModalityHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, D)
            )
        def forward(self, x): return self.net(x)

    heads = nn.ModuleList([ModalityHead() for _ in range(N_MOD)]).to(DEVICE)

    # Step 3: joint fine-tuning
    print("\n  [Router + head fine-tuning (1000 steps)]")
    trainable = (list(router.parameters()) + list(heads.parameters()) +
                 list(backbone.metaplasticity.state_gate.parameters()) +
                 list(backbone.output_proj.parameters()))
    opt = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=1e-4)

    LAMBDA_R   = 1.0
    LAMBDA_NTP = 0.3
    LAMBDA_BAL = 0.05

    data_iter = iter(runner.train_set)
    router_losses, bal_losses, ntp_losses = [], [], []

    for step in range(1, 1001):
        try:
            text, domain_label = next(data_iter)
        except StopIteration:
            data_iter = iter(runner.train_set)
            text, domain_label = next(data_iter)

        opt.zero_grad()

        # ── Single combined backward pass ────────────────────────────────
        # Key fix: ensure partition state is a fresh leaf before NTP forward,
        # preventing the router's cellular_step graph from leaking into the NTP graph.

        # 1. Router classification loss (short prefix)
        enc   = backbone.encode_input(text[:200])
        state = backbone.cellular_step(enc)
        w, logits, bal_loss = router(state.detach(), training=True)
        target = torch.tensor([domain_label], dtype=torch.long, device=DEVICE)
        l_router = F.cross_entropy(logits.unsqueeze(0), target)
        total_loss = LAMBDA_R * l_router + LAMBDA_BAL * bal_loss

        # 2. Reset state to a fresh leaf tensor (detach isolates from router graph)
        with torch.no_grad():
            backbone.partitions._buffers["state"] = torch.zeros_like(
                backbone.partitions._buffers["state"])
        backbone.memory_formation.reset()

        # 3. NTP forward (fresh computation graph, fully isolated from router graph)
        toks = backbone.encoder.tokenize(text)[:256]
        nll_val = 0.0
        if len(toks) >= 2:
            tok_ids = torch.tensor(toks, dtype=torch.long, device=DEVICE)
            seg_len = min(len(toks) - 1, 48)
            embs    = backbone.encoder.embedding(tok_ids[:seg_len+1]) * backbone.encoder._scale
            seg_loss = torch.tensor(0.0, device=DEVICE)
            count = 0
            for t in range(seg_len):
                inp = embs[t]
                backbone.partitions.step(inp)
                agg  = backbone.partitions.aggregate()
                mem  = backbone.memory_formation(inp, agg)
                out  = backbone.metaplasticity(agg, mem, inp)
                st   = backbone.output_proj(out)
                wts, _, _ = router(st.detach(), training=True)
                head_outs = torch.stack([h(st) for h in heads], dim=0)
                routed = (wts.unsqueeze(1) * head_outs).sum(dim=0)
                logits_ntp = routed @ backbone.encoder.embedding.weight.t()
                seg_loss = seg_loss + F.cross_entropy(
                    logits_ntp.unsqueeze(0), tok_ids[t+1].unsqueeze(0))
                count += 1

            if count > 0:
                nll_val = seg_loss.item() / count
                total_loss = total_loss + LAMBDA_NTP * seg_loss / count
                ntp_losses.append(nll_val)

        # 4. Single backward
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        router.step_temperature()

        router_losses.append(l_router.item())
        bal_losses.append(bal_loss.item())

        if step % 200 == 0:
            r_avg   = sum(router_losses[-200:]) / 200
            b_avg   = sum(bal_losses[-200:]) / 200
            n_avg   = sum(ntp_losses[-50:]) / max(len(ntp_losses[-50:]), 1)
            ent     = router.routing_entropy()
            print(f"    step={step:>4}  router={r_avg:.3f}  bal={b_avg:.3f}  "
                  f"ntp={n_avg:.3f}  routing_H={ent:.3f}  T={router.temperature:.2f}")

    # Evaluate routing accuracy
    print("\n  [Routing accuracy eval]")
    backbone.eval()
    W_snap = backbone.metaplasticity.W.data.clone()
    correct, total = 0, 0
    for text, label in random.sample(runner.eval_set, min(300, len(runner.eval_set))):
        backbone.metaplasticity.W.data.copy_(W_snap)
        with torch.no_grad():
            enc   = backbone.encode_input(text[:200])
            state = backbone.cellular_step(enc)
            w, logits, _ = router(state, training=False)
            pred = logits.argmax().item()
        if pred == label:
            correct += 1
        total += 1
    backbone.metaplasticity.W.data.copy_(W_snap)

    acc = correct / total
    print(f"    Routing accuracy: {acc:.3f} (vs 0.333 uniform random)")
    print(f"    Router entropy:   {router.routing_entropy():.3f} (max={math.log(3):.3f})")

    return {
        "name":          "E4_EntropyRouter",
        "backbone_train": bb_result,
        "router_losses": router_losses[::10],
        "bal_losses":    bal_losses[::10],
        "ntp_losses":    ntp_losses[::10],
        "routing_acc":   acc,
        "routing_entropy": router.routing_entropy(),
        "info":          {"n_params": sum(p.numel() for p in backbone.parameters()) +
                                       sum(p.numel() for p in router.parameters()) +
                                       sum(p.numel() for p in heads.parameters())},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from cellai_core.base  import ModelParams
    from v3.cell_ai_v3     import CellAIv3, V3Config

    print("\n--- Loading data ---")
    dataset = build_dataset(n_per=6_000)
    runner  = ExperimentRunner(dataset)

    results = {}

    # ── E0: Fixed baseline (multi-domain, dense PDE, full Hebbian) ────────
    print("\n\n>>> E0: FIXED BASELINE (multi-domain, dense PDE, full Hebbian)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E0_baseline", pde_type="dense", hebbian_type="full",
                      partition_type="single", resonance_type="none")
    m0 = make_model(params, cfg)
    results["E0"] = runner.run_experiment(m0, "E0_Baseline")
    del m0; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── E1: SpectralPDE ───────────────────────────────────────────────────
    print("\n\n>>> E1: SPECTRAL PDE (FFT diffusion, O(D log D))")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E1_spectral_pde", pde_type="spectral", hebbian_type="full",
                      partition_type="single", resonance_type="none")
    m1 = make_model(params, cfg)
    results["E1"] = runner.run_experiment(m1, "E1_SpectralPDE")
    del m1; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── E2: SparseHebbian ─────────────────────────────────────────────────
    print("\n\n>>> E2: SPARSE HEBBIAN (top-k=D/8, O(D log D))")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E2_sparse_hebbian", pde_type="dense", hebbian_type="sparse",
                      partition_type="single", resonance_type="none", k_frac=0.125)
    m2 = make_model(params, cfg)
    results["E2"] = runner.run_experiment(m2, "E2_SparseHebbian")
    del m2; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── E3: MultiScale ────────────────────────────────────────────────────
    print("\n\n>>> E3: MULTI-SCALE PARTITIONS (fast N=4 D=128, slow N=2 D=256, K=8)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E3_multiscale", pde_type="dense", hebbian_type="full",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8)
    m3 = make_model(params, cfg)
    results["E3"] = runner.run_experiment(m3, "E3_MultiScale")
    del m3; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── E4: EntropyRouter ─────────────────────────────────────────────────
    print("\n\n>>> E4: ENTROPY ROUTER (Gumbel-Softmax + load balance)")
    results["E4"] = run_e4_multimodal(runner)
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── E5: Per-frequency Resonance ────────────────────────────────────────
    print("\n\n>>> E5: PER-FREQUENCY RESONANCE (complex FFT filter H∈C^(D/2+1))")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E5_perfreq_resonance", pde_type="dense", hebbian_type="full",
                      partition_type="single", resonance_type="per_freq")
    m5 = make_model(params, cfg)
    results["E5"] = runner.run_experiment(m5, "E5_PerFreqResonance")
    del m5; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── E6: Large scale (D=512, N=8) ──────────────────────────────────────
    print("\n\n>>> E6: LARGE SCALE (D=512, N=8 partitions)")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E6_large_D512_N8", pde_type="dense", hebbian_type="full",
                      partition_type="single", resonance_type="none")
    m6 = make_model(params, cfg)
    results["E6"] = runner.run_experiment(m6, "E6_LargeScale")
    del m6; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── E7: Combined best (SpectralPDE + SparseHebbian + PerFreq + D=512) ─
    print("\n\n>>> E7: COMBINED BEST (Spectral PDE + Sparse Hebbian + PerFreq Resonance, D=512)")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E7_combined_best", pde_type="spectral", hebbian_type="sparse",
                      partition_type="single", resonance_type="per_freq",
                      k_frac=0.125)
    m7 = make_model(params, cfg)
    results["E7"] = runner.run_experiment(m7, "E7_Combined")

    # Longer generation test on E7
    print("\n  [Extended generation test — E7 (32 tokens)]")
    runner.eval_generation(m7, n_prompts=6)
    del m7; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("ARCHITECTURE SEARCH SUMMARY")
    print("="*72)
    print(f"{'Exp':<6} {'Label':<30} {'Init NLL':>9} {'Final NLL':>9} {'PPL(macro)':>12} "
          f"{'ms/call':>8} {'#params':>10}")
    print("-"*72)
    for exp, r in results.items():
        if "train" not in r:
            # E4 multimodal — different format
            r_train = r.get("backbone_train", {})
            ppl_m   = "-"
            ms      = "-"
            n_par   = r.get("info", {}).get("n_params", 0)
        else:
            r_train = r["train"]
            ppl_m   = f"{r['ppl'].get('macro_ppl', 0):>12,.0f}"
            ms      = f"{r['throughput']['full_ms']:.2f}"
            n_par   = r["info"]["n_params"]
        label   = r.get("info", {}).get("label", exp)
        init    = r_train.get("initial", 0)
        final   = r_train.get("final", 0)
        print(f"{exp:<6} {label:<30} {init:>9.2f} {final:>9.2f} {ppl_m} {ms:>8} {n_par:>10,}")

    # Save results
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results -> {out_path}")
    print("DONE.")
    return results


if __name__ == "__main__":
    main()
