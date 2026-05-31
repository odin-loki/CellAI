"""
CellularAI v3 — Guided Architecture Search  Round 2  (E8–E14)
==============================================================
Addresses all weaknesses found in E0-E7:

  E8:  SpectralPDE + MultiScale         — combine the two best D=256 components
  E9:  SpectralPDE + PerFreq (gated)    — fixed dead-init trap with multiplicative gate
  E10: SpectralPDE + SparseHebbian(fix) — deferred Hebbian update, no interference
  E11: SpectralPDE + AdaptiveRouter     — adaptive lambda_bal prevents entropy-accuracy conflict
  E12: Best D=256 combined              — SpectralPDE + MultiScale + PerFreq(gated)
  E13: E12 + generation fixes           — repetition penalty + stochastic injection
  E14: E12 at D=512, 4000 steps         — fair scale comparison

All experiments log to arch_v2.log.
Results saved to data/local/arch_search/results_v2.json.
"""
from __future__ import annotations

import gc, json, math, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = _REPO / "data" / "local"
OUT_DIR  = DATA_DIR / "arch_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_FIELDS = [
    (DATA_DIR / "nlp"  / "train.jsonl", "text",    0),
    (DATA_DIR / "code" / "train.jsonl", "content", 1),
    (DATA_DIR / "math" / "train.jsonl", "problem", 2),
]
MODALITY_NAMES = ["text", "code", "math"]

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print("=" * 72)
print("CELL AI v3 — ARCH SEARCH ROUND 2 (E8-E14)")
print("=" * 72)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Reuse build_dataset and ExperimentRunner from round 1 ────────────────────
from arch_search.run_arch_search import build_dataset, ExperimentRunner

from cellai_core.base    import ModelParams
from cellai_core.routing import AnnealedRouter
from v3.cell_ai_v3       import CellAIv3, V3Config


def make_model(params: ModelParams, cfg: V3Config) -> CellAIv3:
    m = CellAIv3(params, cfg).to(DEVICE)
    n = sum(p.numel() for p in m.parameters())
    cfg.label = cfg.label  # already set
    return m


# ─────────────────────────────────────────────────────────────────────────────
# E11: Adaptive AnnealedRouter
# ─────────────────────────────────────────────────────────────────────────────

def run_e11_adaptive_router(runner: ExperimentRunner) -> Dict:
    """
    E11: SpectralPDE backbone + AnnealedRouter with adaptive lambda_bal.
    lambda decays 0.1 → 0.001 over 1000 steps (phase-2 allows discrimination).
    """
    print("\n" + "=" * 68)
    print("  E11 — Adaptive AnnealedRouter (lambda_bal: 0.1 -> 0.001)")
    print("=" * 68)

    params  = ModelParams(state_size=256, num_partitions=4, device=str(DEVICE))
    cfg_bb  = V3Config(label="E11_adaptive_router", pde_type="spectral",
                       hebbian_type="full", partition_type="single",
                       resonance_type="none")
    backbone = make_model(params, cfg_bb)
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"  Backbone params: {n_params:,}")

    LAMBDA_R   = 0.5
    LAMBDA_BAL = 1.0   # bal_loss already scaled inside AnnealedRouter
    LAMBDA_NTP = 1.0
    n_mod      = len(MODALITY_NAMES)

    router = AnnealedRouter(
        params.state_size, n_modalities=n_mod,
        T_start=2.0, T_end=0.5, anneal_steps=1000,
        lambda_start=0.1, lambda_end=0.001,
    ).to(DEVICE)

    heads = nn.ModuleList([
        nn.Linear(params.state_size, params.state_size, bias=True)
        for _ in range(n_mod)
    ]).to(DEVICE)

    trainable = (list(backbone.parameters()) +
                 list(router.parameters()) +
                 list(heads.parameters()))
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1.5e-5)

    data_iter  = iter(runner.train_set)
    r_losses, b_losses, ntp_losses = [], [], []

    print("  [Backbone + Router joint training (3000 steps)]")
    for step in range(1, 3001):
        try:
            text, domain_label = next(data_iter)
        except StopIteration:
            data_iter = iter(runner.train_set)
            text, domain_label = next(data_iter)

        opt.zero_grad()

        # Fresh partition state
        with torch.no_grad():
            backbone.partitions._buffers["state"] = torch.zeros_like(
                backbone.partitions._buffers["state"])
        backbone.memory_formation.reset()

        # NTP loss through routed heads (48 tokens max)
        toks = backbone.encoder.tokenize(text)[:256]
        nll_val = 0.0
        if len(toks) >= 2:
            tok_ids  = torch.tensor(toks, dtype=torch.long, device=DEVICE)
            seg_len  = min(len(toks) - 1, 48)
            embs     = backbone.encoder.embedding(tok_ids[:seg_len+1]) * backbone.encoder._scale
            seg_loss = torch.tensor(0.0, device=DEVICE)
            count    = 0
            for t in range(seg_len):
                inp = embs[t]
                backbone.partitions.step(inp)
                agg  = backbone.partitions.aggregate()
                mem  = backbone.memory_formation(inp, agg)
                out  = backbone.metaplasticity(agg, mem, inp)
                st   = backbone.output_proj(out)
                wts, _, _ = router(st.detach(), training=True)
                head_outs  = torch.stack([h(st) for h in heads], dim=0)
                routed     = (wts.unsqueeze(1) * head_outs).sum(dim=0)
                logits_ntp = routed @ backbone.encoder.embedding.weight.t()
                seg_loss   = seg_loss + F.cross_entropy(
                    logits_ntp.unsqueeze(0), tok_ids[t+1].unsqueeze(0))
                count += 1
            if count > 0:
                nll_val = seg_loss.item() / count
                ntp_losses.append(nll_val)

        # Router classification loss (short prefix, fresh graph)
        enc        = backbone.encode_input(text[:200])
        cell_state = backbone.cellular_step(enc)
        _, logits_r, bal_loss = router(cell_state.detach(), training=True)
        target     = torch.tensor([domain_label], dtype=torch.long, device=DEVICE)
        l_router   = F.cross_entropy(logits_r.unsqueeze(0), target)

        total_loss = (LAMBDA_NTP * seg_loss / max(count, 1)
                      + LAMBDA_R * l_router
                      + LAMBDA_BAL * router.lambda_bal * bal_loss)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sch.step()
        router.step_temperature()

        r_losses.append(l_router.item())
        b_losses.append(bal_loss.item())

        if step % 300 == 0:
            r_avg  = sum(r_losses[-300:]) / 300
            b_avg  = sum(b_losses[-300:]) / 300
            n_avg  = sum(ntp_losses[-50:]) / max(len(ntp_losses[-50:]), 1)
            ent    = router.routing_entropy()
            print(f"    step={step:>4}  router={r_avg:.3f}  bal={b_avg:.3f}  "
                  f"ntp={n_avg:.3f}  H={ent:.3f}  T={router.temperature:.2f}  "
                  f"lam={router.lambda_bal:.4f}")

    # Routing accuracy eval
    print("  [Routing accuracy eval]")
    correct, total = 0, 0
    with torch.no_grad():
        for text, label in runner.train_set[:300]:
            enc   = backbone.encode_input(text[:200])
            state = backbone.cellular_step(enc)
            _, logits_r, _ = router(state, training=False)
            pred = logits_r.argmax().item()
            if pred == label:
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    ent = router.routing_entropy()
    print(f"    Routing accuracy: {acc:.3f}  (random={1/n_mod:.3f})")
    print(f"    Router entropy:   {ent:.3f}  (max={math.log(n_mod):.3f})")

    return {
        "info": {"label": "E11_adaptive_router", "n_params": n_params + sum(p.numel() for p in router.parameters()) + sum(p.numel() for p in heads.parameters())},
        "router": {"accuracy": acc, "entropy": ent},
        "train": {"final_ntp": sum(ntp_losses[-100:]) / max(len(ntp_losses[-100:]), 1)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    print("\n--- Loading data ---")
    dataset = build_dataset(n_per=6_000)
    runner  = ExperimentRunner(dataset)

    # ── E8: SpectralPDE + MultiScale ──────────────────────────────────────────
    print("\n\n>>> E8: SPECTRAL PDE + MULTI-SCALE (best of E1 + E3)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E8_spectral_multiscale",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8)
    m8 = make_model(params, cfg)
    results["E8"] = runner.run_experiment(m8, "E8_SpectralMultiScale")
    del m8; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E9: SpectralPDE + Fixed PerFreqResonance (multiplicative gate) ────────
    print("\n\n>>> E9: SPECTRAL PDE + PER-FREQ RESONANCE (multiplicative gate FIX)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E9_spectral_perfreq_gated",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="single", resonance_type="per_freq")
    m9 = make_model(params, cfg)
    results["E9"] = runner.run_experiment(m9, "E9_SpectralPerFreqGated")
    del m9; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E10: SpectralPDE + SparseHebbian (deferred update, no interference) ───
    print("\n\n>>> E10: SPECTRAL PDE + SPARSE HEBBIAN (deferred update FIX)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E10_spectral_sparse_deferred",
                      pde_type="spectral", hebbian_type="sparse",
                      partition_type="single", resonance_type="none", k_frac=0.125)
    m10 = make_model(params, cfg)
    results["E10"] = runner.run_experiment(m10, "E10_SpectralSparseFixed")
    del m10; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E11: AdaptiveRouter (lambda annealing) ────────────────────────────────
    print("\n\n>>> E11: ADAPTIVE ANNEALED ROUTER (lambda_bal: 0.1 -> 0.001)")
    results["E11"] = run_e11_adaptive_router(runner)
    gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E12: Best D=256 combined (Spectral + MultiScale + PerFreq gated) ──────
    print("\n\n>>> E12: BEST D=256 COMBINED (SpectralPDE + MultiScale + PerFreq gated)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E12_best_d256",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="per_freq",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8,
                      gen_rep_penalty=1.3, gen_noise_std=0.03)
    m12 = make_model(params, cfg)
    results["E12"] = runner.run_experiment(m12, "E12_BestD256")

    # ── E13: Generation quality evaluation on E12 ─────────────────────────────
    print("\n\n>>> E13: GENERATION QUALITY TEST (E12 + rep_penalty + noise)")
    print("  [Extended generation with fixed decoding — 6 prompts × 64 tokens]")
    prompts = [
        ("math", "Solve for x: 2x + 5 = 13"),
        ("math", "The derivative of x squared is"),
        ("code", "def fibonacci(n):\n    if n <= 1: return n\n    return"),
        ("code", "class Node:\n    def __init__(self, val):"),
        ("text", "The transformer architecture learns"),
        ("text", "Neural networks are trained by"),
    ]
    gen_results = []
    for domain, prompt in prompts:
        t0  = time.perf_counter()
        gen = m12.generate(prompt, max_tokens=48, temperature=0.85,
                           rep_penalty=1.3, noise_std=0.03)
        ms  = (time.perf_counter() - t0) * 1000
        continuation = gen[len(prompt):]
        # Count unique tokens as diversity metric
        toks = m12.encoder.tokenize(continuation)
        unique_frac = len(set(toks)) / max(len(toks), 1)
        print(f"    [{domain}] '{prompt[:40]}'")
        print(f"           -> '{continuation[:60]}'  [{ms:.0f}ms, diversity={unique_frac:.2f}]")
        gen_results.append({"domain": domain, "prompt": prompt,
                            "continuation": continuation,
                            "diversity": unique_frac, "ms": ms})
    results["E13"] = {"generation": gen_results,
                      "mean_diversity": sum(g["diversity"] for g in gen_results) / len(gen_results)}
    avg_div = results["E13"]["mean_diversity"]
    print(f"\n  Mean token diversity: {avg_div:.3f}  (1.0=all unique, lower=more repetitive)")

    del m12; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E14: E12 at D=512, 4000 steps (fair scale comparison) ─────────────────
    print("\n\n>>> E14: BEST RECIPE AT D=512 (4000 steps — fair vs E7's 2000 steps)")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E14_best_d512_4k",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="per_freq",
                      N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                      gen_rep_penalty=1.3, gen_noise_std=0.03)
    m14 = make_model(params, cfg)

    # Run with 4000 steps (double the standard 2000)
    results["E14"] = runner.run_experiment(m14, "E14_BestD512_4k", n_steps=4000)

    print("\n  [E14 Generation — 4000-step D=512 model]")
    for domain, prompt in prompts[:4]:
        gen = m14.generate(prompt, max_tokens=48, temperature=0.85)
        toks = m14.encoder.tokenize(gen[len(prompt):])
        div  = len(set(toks)) / max(len(toks), 1)
        print(f"    [{domain}] '{prompt[:40]}' -> '{gen[len(prompt):len(prompt)+60]}'  [div={div:.2f}]")

    del m14; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ROUND 2 ARCHITECTURE SEARCH SUMMARY (E8-E14)")
    print("=" * 72)
    print(f"{'Exp':<6} {'Label':<35} {'Init':>8} {'Final':>8} {'PPL(mac)':>14} {'ms/call':>8}")
    print("-" * 72)
    for exp in ["E8","E9","E10","E11","E12","E14"]:
        r = results.get(exp, {})
        if not r:
            continue
        tr   = r.get("train", {})
        ppl  = r.get("ppl",{}).get("macro_ppl", "-")
        ms   = r.get("throughput", {}).get("full_ms", "-")
        init = tr.get("initial", 0)
        fin  = tr.get("final", 0)
        lbl  = r.get("info",{}).get("label", exp)
        if isinstance(ppl, float): ppl = f"{ppl:.1f}"
        if isinstance(ms, float):  ms  = f"{ms:.3f}"
        print(f"{exp:<6} {str(lbl):<35} {init:>8.1f} {fin:>8.3f} {str(ppl):>14} {str(ms):>8}")

    # E13 generation results
    e13 = results.get("E13", {})
    print(f"\nE13 mean generation diversity: {e13.get('mean_diversity', 0):.3f}")

    # Save results
    results_path = OUT_DIR / "results_v2.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved -> {results_path}")
    print("DONE.")
    return results


if __name__ == "__main__":
    main()
