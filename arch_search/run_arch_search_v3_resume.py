"""Resume arch search v3 from E17 (after fixing partition reset bug)."""
from __future__ import annotations

import gc, json, math, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = _REPO / "data" / "local"
OUT_DIR  = DATA_DIR / "arch_search"

MODALITY_NAMES  = ["text", "code", "math"]
DOMAIN_PREFIXES = {0: "<<TEXT>> ", 1: "<<CODE>> ", 2: "<<MATH>> "}

random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True

print("=" * 72)
print("CELL AI v3 — ARCH SEARCH ROUND 3 RESUME (E17-E20)")
print("=" * 72)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from arch_search.run_arch_search import build_dataset, ExperimentRunner
from arch_search.run_arch_search_v3 import build_domain_dataset, run_with_routing

from cellai_core.base import ModelParams
from v3.cell_ai_v3   import CellAIv3, V3Config


def make_model(params, cfg):
    return CellAIv3(params, cfg).to(DEVICE)


def main():
    results = {}
    if (OUT_DIR / "results_v3.json").exists():
        with open(OUT_DIR / "results_v3.json") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    print("\n--- Loading data ---")
    plain_dataset         = build_dataset(n_per=6_000)
    runner                = ExperimentRunner(plain_dataset)
    dom_train, dom_eval   = build_domain_dataset(n_per=6_000)
    print(f"  Domain train: {len(dom_train)}  eval: {len(dom_eval)}")

    # ── E17: Domain-token routing (fixed partition reset) ─────────────────────
    print("\n\n>>> E17: DOMAIN-TOKEN ROUTING (fixed)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E17_domain_token_routing",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8)
    m17 = make_model(params, cfg)

    # Train backbone on plain NTP first (2000 steps)
    train_r17 = runner.train(m17)
    print(f"  Backbone pre-training: init={train_r17['initial']:.1f}  final={train_r17['final']:.3f}")

    # Now fine-tune on domain-tagged data (500 steps)
    fine_tune_runner = ExperimentRunner(dom_train + dom_eval)
    fine_tune_runner.TRAIN_STEPS = 500
    fine_tune_runner.LR = 1e-4
    ft_r = fine_tune_runner.train(m17)
    print(f"  Domain fine-tune: final={ft_r['final']:.3f}")

    # Test routing
    r17 = run_with_routing(m17, dom_train, dom_eval, n_steps=2000, label="E17_routing")
    results["E17"] = {"backbone_train": train_r17, "finetune": ft_r, **r17}
    del m17; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E18: SpectralPDE + MultiScale D=512, 8000 steps ───────────────────────
    print("\n\n>>> E18: SPECTRAL + MULTISCALE D=512, 8000 STEPS")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E18_spectral_multi_d512_8k",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                      gen_rep_penalty=1.3, gen_noise_std=0.03)
    m18 = make_model(params, cfg)
    print(f"  Parameters: {sum(p.numel() for p in m18.parameters()):,}")
    results["E18"] = runner.run_experiment(m18, "E18_SpectralMultiD512_8k", n_steps=8000)

    print("  [E18 generation]")
    for p in ["Solve for x: 2x + 5 = 13", "def fibonacci(n):", "The transformer learns"]:
        gen = m18.generate(p, max_tokens=40, temperature=0.85)
        cont = gen[len(p):]
        div = len(set(m18.encoder.tokenize(cont))) / max(len(m18.encoder.tokenize(cont)), 1)
        print(f"    '{p[:35]}' -> '{cont[:55]}'  [div={div:.2f}]")
    del m18; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E19: Best D=256 (E15+E16 combined: Spectral+Multi+Sparse+Resonance) ───
    print("\n\n>>> E19: BEST D=256 — SPECTRAL+MULTI+SPARSE+RESONANCE (all fixes)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E19_best_d256_all_fixes",
                      pde_type="spectral", hebbian_type="sparse",
                      partition_type="multiscale", resonance_type="per_freq",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8,
                      k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
    m19 = make_model(params, cfg)
    print(f"  Parameters: {sum(p.numel() for p in m19.parameters()):,}")
    results["E19"] = runner.run_experiment(m19, "E19_BestD256AllFixes")

    r19 = run_with_routing(m19, dom_train, dom_eval, n_steps=1000, label="E19_routing")
    results["E19_routing"] = r19
    del m19; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E20: Champion — Spectral+Multi+Sparse D=512, 8000 steps ─────────────
    print("\n\n>>> E20: CHAMPION — SPECTRAL+MULTI+SPARSE D=512, 8000 STEPS")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E20_champion_sparse_d512_8k",
                      pde_type="spectral", hebbian_type="sparse",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                      k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
    m20 = make_model(params, cfg)
    n20 = sum(p.numel() for p in m20.parameters())
    print(f"  Parameters: {n20:,}")
    results["E20"] = runner.run_experiment(m20, "E20_Champion_SparseD512_8k", n_steps=8000)

    # E20 generation showcase
    print("\n  [E20 FINAL GENERATION SHOWCASE]")
    prompts = [
        ("math",  "Solve for x: 3x^2 - 12 = 0"),
        ("math",  "Integrate x^2 from 0 to 1"),
        ("code",  "def binary_search(arr, target):"),
        ("code",  "class LinkedList:\n    def __init__(self):"),
        ("text",  "The cellular automaton model predicts that"),
        ("text",  "Neural networks learn representations by"),
    ]
    gen_results = []
    for domain, prompt in prompts:
        gen  = m20.generate(prompt, max_tokens=48, temperature=0.85)
        cont = gen[len(prompt):]
        toks = m20.encoder.tokenize(cont)
        div  = len(set(toks)) / max(len(toks), 1)
        print(f"    [{domain}] '{prompt[:40]}'")
        print(f"           -> '{cont[:60]}'  [diversity={div:.2f}]")
        gen_results.append({"domain": domain, "prompt": prompt,
                            "continuation": cont, "diversity": div})
    results["E20_generation"] = gen_results

    del m20; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ROUND 3 SUMMARY (E15-E20)")
    print("=" * 72)
    print(f"{'Exp':<6} {'Label':<38} {'Init':>8} {'Final':>8} {'MacroPPL':>12}")
    print("-" * 72)
    for exp in ["E15","E16","E18","E19","E20"]:
        r = results.get(exp, {})
        if not r: continue
        tr  = r.get("train", {})
        ppl = r.get("ppl",{}).get("macro_ppl", "—")
        ini = tr.get("initial", 0)
        fin = tr.get("final", 0)
        lbl = r.get("info",{}).get("label", exp)
        if isinstance(ppl, float): ppl = f"{ppl:.1f}"
        print(f"{exp:<6} {str(lbl):<38} {ini:>8.1f} {fin:>8.3f} {str(ppl):>12}")

    for exp_r in ["E17","E19_routing"]:
        r = results.get(exp_r, {})
        if r:
            acc = r.get("routing",{}).get("accuracy","—")
            ent = r.get("routing",{}).get("entropy","—")
            ntp = r.get("ntp_final","—")
            if isinstance(acc, float): acc = f"{acc:.3f}"
            if isinstance(ent, float): ent = f"{ent:.3f}"
            if isinstance(ntp, float): ntp = f"{ntp:.3f}"
            print(f"{exp_r:<6} routing_acc={acc}  entropy={ent}  ntp={ntp}")

    results_path = OUT_DIR / "results_v3.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results -> {results_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
