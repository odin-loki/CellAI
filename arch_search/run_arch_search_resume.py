"""
Resume arch search from E4 (E0-E3 already have checkpoints).
"""
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

MODALITY_FIELDS = [
    (DATA_DIR / "nlp"  / "train.jsonl",  "text",    0),
    (DATA_DIR / "code" / "train.jsonl",  "content", 1),
    (DATA_DIR / "math" / "train.jsonl",  "problem", 2),
]
MODALITY_NAMES = ["text", "code", "math"]

random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print("=" * 72)
print("CELL AI v3 — ARCH SEARCH RESUME (E4-E7)")
print("=" * 72)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Import from main script ───────────────────────────────────────────────────
from arch_search.run_arch_search import (
    build_dataset, ExperimentRunner, run_e4_multimodal, make_model
)

# Load previous results
results_path = OUT_DIR / "results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded existing results: {list(results.keys())}")
else:
    results = {}


def main():
    from cellai_core.base  import ModelParams
    from v3.cell_ai_v3     import CellAIv3, V3Config

    print("\n--- Loading data ---")
    dataset = build_dataset(n_per=6_000)
    runner  = ExperimentRunner(dataset)

    # ── E4: EntropyRouter ─────────────────────────────────────────────────
    print("\n\n>>> E4: ENTROPY ROUTER (Gumbel-Softmax + load balance)")
    results["E4"] = run_e4_multimodal(runner)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── E5: Per-frequency Resonance ────────────────────────────────────────
    print("\n\n>>> E5: PER-FREQUENCY RESONANCE (complex FFT filter H∈C^(D/2+1))")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E5_perfreq_resonance", pde_type="dense", hebbian_type="full",
                      partition_type="single", resonance_type="per_freq")
    m5 = make_model(params, cfg)
    results["E5"] = runner.run_experiment(m5, "E5_PerFreqResonance")
    del m5
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── E6: Large scale (D=512, N=8) ──────────────────────────────────────
    print("\n\n>>> E6: LARGE SCALE (D=512, N=8 partitions)")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E6_large_D512_N8", pde_type="dense", hebbian_type="full",
                      partition_type="single", resonance_type="none")
    m6 = make_model(params, cfg)
    results["E6"] = runner.run_experiment(m6, "E6_LargeScale")
    del m6
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── E7: Combined best (SpectralPDE + SparseHebbian + PerFreq + D=512) ─
    print("\n\n>>> E7: COMBINED BEST (Spectral PDE + Sparse Hebbian + PerFreq Resonance, D=512)")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E7_combined_best", pde_type="spectral", hebbian_type="sparse",
                      partition_type="single", resonance_type="per_freq",
                      k_frac=0.125)
    m7 = make_model(params, cfg)
    results["E7"] = runner.run_experiment(m7, "E7_Combined")

    print("\n  [Extended generation test — E7]")
    runner.eval_generation(m7, n_prompts=6)
    del m7
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL ARCHITECTURE SEARCH SUMMARY (all 8 experiments)")
    print("=" * 72)
    print(f"{'Exp':<6} {'Label':<30} {'Init':>8} {'Final':>8} {'PPL(mac)':>12} {'ms/call':>8} {'#params':>10}")
    print("-" * 72)
    for exp in ["E0","E1","E2","E3","E4","E5","E6","E7"]:
        r = results.get(exp, {})
        if not r:
            continue
        if "train" in r:
            init  = r["train"].get("initial", 0)
            final = r["train"].get("final", 0)
            ppl   = r["ppl"].get("macro_ppl", 0)
            ms    = r["throughput"]["full_ms"]
            nparm = r["info"]["n_params"]
        else:
            # E4 format
            bt = r.get("backbone_train", {})
            init  = bt.get("initial", 0)
            final = bt.get("final", 0)
            ppl   = "-"
            ms    = "-"
            nparm = r.get("info", {}).get("n_params", 0)
        label = r.get("info", {}).get("label", exp)
        print(f"{exp:<6} {label:<30} {init:>8.1f} {final:>8.3f} {str(ppl):>12} {str(ms):>8} {nparm:>10,}")

    # Save
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved -> {results_path}")
    print("DONE.")
    return results


if __name__ == "__main__":
    main()
