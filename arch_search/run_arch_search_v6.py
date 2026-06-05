"""
CellularAI v3 — Architecture Search Wave 3  (E30–E32)
=======================================================
Phase C hypotheses vs E27 reset champion (text+code NLL 7.847):

  E30: Encoder→state bypass pathway (resume E27 + 4k fine-tune)
  E31: Input-side domain PDE banks on slow spectral partitions (+ 4k)
  E32: PerFreqResonance at D=1024 (resume E27 + 4k fine-tune)

Run:
    python -m arch_search.run_arch_search_v6 --train E30
    python -m arch_search.run_arch_search_v6 --train E30 E31 E32
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arch_search.eval_protocol import attach_protocol, load_eval_protocol
from arch_search.harness import RunMeta, OUT_DIR, save_results_atomic, write_meta_json
from arch_search.run_arch_search import ExperimentRunner, build_dataset
from arch_search.run_arch_search_v4 import eval_ppl_continuous, gen_showcase, make_model
from arch_search.run_arch_search_v5 import eval_with_protocol, headline_value, load_checkpoint
from cellai_core.base import ModelParams
from v3.cell_ai_v3 import CellAIv3, V3Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = OUT_DIR / "results_v6.json"
CHAMPION_CKPT = OUT_DIR / "E27_D1024_16k.pt"
CHAMPION_TC_NLL = 7.847


def _base_d1024_cfg(label: str, **extra) -> Tuple[ModelParams, V3Config]:
    params = ModelParams(state_size=1024, num_partitions=8)
    cfg_kw = dict(
        label=label,
        pde_type="spectral",
        hebbian_type="sparse",
        partition_type="multiscale",
        resonance_type="none",
        N_fast=8,
        D_fast=256,
        N_slow=4,
        D_slow=512,
        K_slow=8,
        k_frac=0.125,
        gen_rep_penalty=1.3,
        gen_noise_std=0.03,
    )
    cfg_kw.update(extra)
    cfg = V3Config(**cfg_kw)
    return params, cfg


def build_e30() -> Tuple[CellAIv3, int]:
    params, cfg = _base_d1024_cfg("E30_bypass_d1024", use_bypass=True)
    m = make_model(params, cfg)
    return m, sum(p.numel() for p in m.parameters())


def build_e31() -> Tuple[CellAIv3, int]:
    params, cfg = _base_d1024_cfg("E31_domain_banks_d1024", use_domain_banks=True)
    m = make_model(params, cfg)
    return m, sum(p.numel() for p in m.parameters())


def build_e32() -> Tuple[CellAIv3, int]:
    params, cfg = _base_d1024_cfg("E32_perfreq_d1024", resonance_type="per_freq")
    m = make_model(params, cfg)
    return m, sum(p.numel() for p in m.parameters())


def resume_from_e27(model: nn.Module) -> int:
    if not CHAMPION_CKPT.exists():
        raise FileNotFoundError(f"Missing champion checkpoint: {CHAMPION_CKPT}")
    sd = torch.load(str(CHAMPION_CKPT), map_location=DEVICE)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded E27 (strict=False): missing={len(missing)} unexpected={len(unexpected)}")
    return len(missing)


def eval_routing_accuracy(
    model: CellAIv3,
    eval_set: List[Tuple[str, int]],
    n: int = 150,
) -> Dict:
    """Held-out accuracy of input-side domain router (E31)."""
    if model.domain_router is None:
        return {"note": "no domain router"}
    model.eval()
    correct, total = 0, 0
    per_label: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    samples = {label: [t for t, l in eval_set if l == label][: n // 3] for label in range(3)}
    with torch.no_grad():
        for label, texts in samples.items():
            for text in texts:
                toks = model.encoder.tokenize(text)[:64]
                if not toks:
                    continue
                tok_ids = torch.tensor(toks, dtype=torch.long, device=DEVICE)
                embs = model.encoder.embedding(tok_ids) * model.encoder._scale
                preds = []
                for t in range(min(8, embs.shape[0])):
                    w = model.domain_router(embs[t])
                    preds.append(int(w.argmax().item()))
                if preds:
                    maj = max(set(preds), key=preds.count)
                    per_label[label].append(float(maj == label))
                    correct += int(maj == label)
                    total += 1
    acc = correct / max(total, 1)
    return {
        "routing_accuracy": acc,
        "n_samples": total,
        "per_label_acc": {
            k: sum(v) / len(v) if v else 0.0 for k, v in per_label.items()
        },
        "chance": 1.0 / 3.0,
    }


def run_wave3_exp(
    exp_id: str,
    model: CellAIv3,
    runner: ExperimentRunner,
    eval_set: List[Tuple[str, int]],
    results: Dict,
    train_steps: int = 4_000,
    lr: float = 5e-5,
) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    proto = load_eval_protocol("reset")

    orig_steps, orig_lr = runner.TRAIN_STEPS, runner.LR
    runner.TRAIN_STEPS = train_steps
    runner.LR = lr
    train_r = runner.train(model)
    runner.TRAIN_STEPS, runner.LR = orig_steps, orig_lr

    ppl = eval_with_protocol(model, eval_set, proto)
    hl = headline_value(ppl)
    print(f"  text+code NLL={hl:.3f}  (champion {CHAMPION_TC_NLL:.3f})")

    routing = eval_routing_accuracy(model, eval_set) if exp_id == "E31" else {}
    if routing:
        print(f"  routing acc={routing.get('routing_accuracy', 0):.3f}  (chance 0.333)")

    gen_r = gen_showcase(model, exp_id)
    ckpt_out = OUT_DIR / f"{exp_id}_{model.cfg.label}.pt"
    torch.save({"state_dict": model.state_dict(), "train": train_r, "ppl": ppl}, ckpt_out)

    meta = RunMeta(
        experiment_id=exp_id,
        hypothesis=f"Wave3 {exp_id} vs E27 champion",
        protocol_id=proto.protocol_id,
        protocol_hash=proto.protocol_hash,
        regime="reset",
        train_steps=train_steps,
        lr=lr,
        resume_from="E27_D1024_16k.pt",
        headline_value=hl,
        extra={"champion_text_code_nll": CHAMPION_TC_NLL},
    )
    write_meta_json(meta, ckpt_out.stem)

    entry = {
        "info": {"label": model.cfg.label, "n_params": n_params, "resume_from": "E27"},
        "train": train_r,
        "ppl": ppl,
        "generation": gen_r,
        "meta": meta.__dict__,
    }
    if hl is not None and hl > CHAMPION_TC_NLL + 1.0:
        entry["failure_note"] = (
            "Catastrophic regression vs E27; likely bypass fine-tune destabilised backbone. "
            "Retry with zero-init bypass (log_alpha=-8, zero bypass_proj)."
        )
    if routing:
        entry["routing"] = routing
    results[exp_id] = entry
    save_results_atomic(RESULTS_PATH, results)

    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wave 3 architecture search (E30–E32)")
    p.add_argument("--train", nargs="+", metavar="EXP", help="E30, E31, and/or E32")
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = _parse_args()
    print("=" * 72)
    print("CELL AI v3 — ARCH SEARCH WAVE 3 (E30–E32)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Champion: E27 text+code NLL = {CHAMPION_TC_NLL}")

    if not args.train:
        print("\nUsage: python -m arch_search.run_arch_search_v6 --train E30")
        return

    results: Dict = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            results = json.load(fh)

    print("\n--- Loading data ---")
    dataset = build_dataset(n_per=6_000)
    runner = ExperimentRunner(dataset)
    eval_set = dataset[int(len(dataset) * 0.85) :]

    only: Set[str] = set()
    for x in args.train:
        s = x.strip().upper()
        if not s.startswith("E"):
            s = "E" + s
        only.add(s)

    builders = {
        "E30": build_e30,
        "E31": build_e31,
        "E32": build_e32,
    }

    for exp in ("E30", "E31", "E32"):
        if exp not in only:
            continue
        print(f"\n>>> {exp}: resume E27 + 4k fine-tune (LR=5e-5)")
        model, _ = builders[exp]()
        resume_from_e27(model)
        run_wave3_exp(exp, model, runner, eval_set, results)

    print("\n" + "=" * 72)
    print("WAVE 3 SUMMARY")
    print("=" * 72)
    print(f"{'Exp':<6} {'text+code NLL':>14} {'vs E27':>10}")
    print("-" * 34)
    for exp in ("E30", "E31", "E32"):
        r = results.get(exp, {})
        tc = r.get("ppl", {}).get("text_code_avg_nll", "—")
        if isinstance(tc, float):
            delta = tc - CHAMPION_TC_NLL
            print(f"{exp:<6} {tc:>14.3f} {delta:>+10.3f}")
        else:
            print(f"{exp:<6} {str(tc):>14} {'—':>10}")
    print(f"\nResults -> {RESULTS_PATH}")
    print("DONE.")


if __name__ == "__main__":
    main()
