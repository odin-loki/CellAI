"""
CellularAI v3 — Wave 4  (E33–E35, hyperband, diagnostics)
==========================================================
  E33: E27 resume +16k (32k total, LR=3e-5, early-stop eval)
  E34: Bypass retry — zero-init, freeze backbone 1k then 3k full fine-tune
  E35: Document alignment aux loss (resume E27 +4k)
  --hyperband: LR screen 600 steps × 3 configs
  --decode-sweep E27: rep_penalty / noise grid on champion
  --ood-eval: held-out slice seed=99 quick eval

  python -m arch_search.run_arch_search_v7 --train E33 E34 E35
  python -m arch_search.run_arch_search_v7 --hyperband
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
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
from arch_search.run_arch_search_v5 import eval_with_protocol, headline_value
from arch_search.run_arch_search_v6 import (
    CHAMPION_CKPT,
    CHAMPION_TC_NLL,
    _base_d1024_cfg,
    resume_from_e27,
)
from arch_search.training_v7 import TrainHarness, freeze_except, unfreeze_all
from cellai_core.base import ModelParams
from v3.cell_ai_v3 import CellAIv3, V3Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = OUT_DIR / "results_v7.json"


def _eval_quick(model: CellAIv3, eval_set: List[Tuple[str, int]], n: int = 30) -> Dict:
    proto = load_eval_protocol("reset")
    ppl = eval_ppl_continuous(model, eval_set, n=n, warm_tokens=0)
    return attach_protocol(ppl, proto)


def _save_exp(
    results: Dict,
    exp_id: str,
    model: CellAIv3,
    train_r: Dict,
    ppl: Dict,
    gen_r: List,
    hypothesis: str,
    resume_from: str,
    train_steps: int,
    lr: float,
    extra: Optional[Dict] = None,
) -> None:
    hl = headline_value(ppl)
    proto = load_eval_protocol("reset")
    ckpt = OUT_DIR / f"{exp_id}_{model.cfg.label}.pt"
    torch.save({"state_dict": model.state_dict(), "train": train_r, "ppl": ppl}, ckpt)
    meta = RunMeta(
        experiment_id=exp_id,
        hypothesis=hypothesis,
        protocol_id=proto.protocol_id,
        protocol_hash=proto.protocol_hash,
        regime="reset",
        train_steps=train_steps,
        lr=lr,
        resume_from=resume_from,
        headline_value=hl,
        extra=extra or {},
    )
    write_meta_json(meta, ckpt.stem)
    entry = {
        "info": {"label": model.cfg.label, "n_params": sum(p.numel() for p in model.parameters())},
        "train": train_r,
        "ppl": ppl,
        "generation": gen_r,
        "meta": meta.__dict__,
    }
    if train_r.get("best_ckpt"):
        entry["checkpoints"] = {"best": train_r["best_ckpt"], "last": train_r["last_ckpt"]}
    results[exp_id] = entry
    save_results_atomic(RESULTS_PATH, results)
    print(f"  text+code NLL={hl:.3f}  vs champion {CHAMPION_TC_NLL:.3f}")


def run_e33(eval_set, train_set, results: Dict) -> None:
    print("\n>>> E33: E27 +16k (32k total, LR=3e-5, eval every 800)")
    params, cfg = _base_d1024_cfg("E33_d1024_32k")
    model = make_model(params, cfg)
    resume_from_e27(model)
    harness = TrainHarness(
        train_set,
        eval_fn=lambda m: _eval_quick(m, eval_set, n=30),
        headline_fn=headline_value,
        eval_every=800,
        early_stop_patience=3,
    )
    train_r = harness.train(model, n_steps=16_000, lr=3e-5, exp_id="E33")
    ppl = _eval_quick(model, eval_set, n=150)
    gen_r = gen_showcase(model, "E33")
    _save_exp(results, "E33", model, train_r, ppl, gen_r,
              "E27 +16k; beat 7.847 text+code", "E27_D1024_16k.pt", 16000, 3e-5)
    del model; gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_e34(eval_set, train_set, results: Dict) -> None:
    print("\n>>> E34: bypass zero-init — freeze 1k then full 3k")
    params, cfg = _base_d1024_cfg("E34_bypass_zero_d1024", use_bypass=True)
    model = make_model(params, cfg)
    resume_from_e27(model)
    freeze_except(model, ("bypass_proj", "log_alpha_bypass"))
    harness = TrainHarness(train_set, eval_fn=lambda m: {}, headline_fn=lambda x: None)
    print("  Phase A: bypass-only 1000 steps")
    train_a = harness.train(model, n_steps=1_000, lr=1e-4, exp_id="E34a")
    unfreeze_all(model)
    harness_b = TrainHarness(
        train_set,
        eval_fn=lambda m: _eval_quick(m, eval_set, n=30),
        headline_fn=headline_value,
        eval_every=600,
    )
    print("  Phase B: full model 3000 steps")
    train_b = harness_b.train(model, n_steps=3_000, lr=5e-5, exp_id="E34")
    train_r = {**train_a, "phase_b": train_b}
    ppl = _eval_quick(model, eval_set, n=150)
    gen_r = gen_showcase(model, "E34")
    _save_exp(results, "E34", model, train_r, ppl, gen_r,
              "Bypass zero-init + staged unfreeze", "E27_D1024_16k.pt", 4000, 5e-5,
              extra={"phase_a_steps": 1000, "phase_b_steps": 3000})
    del model; gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_e35(eval_set, train_set, results: Dict) -> None:
    print("\n>>> E35: doc-align aux loss (resume E27 +4k)")
    params, cfg = _base_d1024_cfg("E35_doc_align_d1024", use_doc_align=True, lambda_doc=0.05)
    model = make_model(params, cfg)
    resume_from_e27(model)
    harness = TrainHarness(train_set, eval_fn=lambda m: {}, headline_fn=lambda x: None)
    train_r = harness.train(model, n_steps=4_000, lr=5e-5, exp_id="E35")
    ppl = _eval_quick(model, eval_set, n=150)
    gen_r = gen_showcase(model, "E35")
    _save_exp(results, "E35", model, train_r, ppl, gen_r,
              "Token-state cosine alignment auxiliary", "E27_D1024_16k.pt", 4000, 5e-5)
    del model; gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_hyperband(eval_set, train_set, results: Dict) -> None:
    print("\n>>> HYPERBAND: LR screen 600 steps × 3")
    configs = [("H1", 1e-4), ("H2", 3e-4), ("H3", 5e-5)]
    hb: Dict = {}
    for hid, lr in configs:
        print(f"\n  --- {hid} LR={lr}")
        params, cfg = _base_d1024_cfg(f"hyperband_{hid}")
        model = make_model(params, cfg)
        resume_from_e27(model)
        harness = TrainHarness(train_set, eval_fn=lambda m: {}, headline_fn=lambda x: None)
        train_r = harness.train(model, n_steps=600, lr=lr, exp_id=hid)
        ppl = _eval_quick(model, eval_set, n=30)
        hl = headline_value(ppl)
        hb[hid] = {"lr": lr, "train": train_r, "ppl": ppl, "text_code_nll": hl}
        print(f"    {hid} text+code NLL={hl:.3f}")
        del model; gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    results["hyperband"] = hb
    best = min(hb.items(), key=lambda x: x[1].get("text_code_nll") or 999)
    results["hyperband_best"] = {"id": best[0], "lr": best[1]["lr"], "nll": best[1]["text_code_nll"]}
    save_results_atomic(RESULTS_PATH, results)


def run_decode_sweep(results: Dict) -> None:
    print("\n>>> DECODE SWEEP on E27 champion")
    if not CHAMPION_CKPT.exists():
        print("  missing E27 checkpoint")
        return
    params, cfg = _base_d1024_cfg("E27_decode_sweep")
    model = make_model(params, cfg)
    model.load_state_dict(torch.load(str(CHAMPION_CKPT), map_location=DEVICE)["state_dict"])
    model.eval()
    prompt = "The transformer architecture learns"
    grid = []
    for rep in (1.0, 1.3, 1.5):
        for noise in (0.0, 0.03, 0.06):
            gen = model.generate(prompt, max_tokens=32, temperature=0.85,
                                 rep_penalty=rep, noise_std=noise)
            cont = gen[len(prompt):]
            toks = model.encoder._tok.encode(cont, disallowed_special=())
            div = len(set(toks)) / max(len(toks), 1)
            grid.append({"rep_penalty": rep, "noise_std": noise,
                         "continuation": cont[:80], "diversity": div})
            print(f"  rep={rep} noise={noise} div={div:.2f}")
    results["decode_sweep_E27"] = grid
    save_results_atomic(RESULTS_PATH, results)


def run_ood_eval(eval_set, results: Dict) -> None:
    print("\n>>> OOD eval E27 (eval subset seed=99)")
    random.seed(99)
    if not CHAMPION_CKPT.exists():
        return
    params, cfg = _base_d1024_cfg("E27_ood")
    model = make_model(params, cfg)
    model.load_state_dict(torch.load(str(CHAMPION_CKPT), map_location=DEVICE)["state_dict"])
    ood = list(eval_set)
    random.shuffle(ood)
    ppl = _eval_quick(model, ood[:90], n=90)
    results["E27_ood_eval"] = ppl
    save_results_atomic(RESULTS_PATH, results)
    print(f"  OOD text+code NLL={headline_value(ppl):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Wave 4 arch search")
    ap.add_argument("--train", nargs="+", metavar="EXP")
    ap.add_argument("--hyperband", action="store_true")
    ap.add_argument("--decode-sweep", action="store_true")
    ap.add_argument("--ood-eval", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("WAVE 4 — E33–E35 + diagnostics")
    print("=" * 72)

    results: Dict = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            results = json.load(fh)

    dataset = build_dataset(n_per=6_000)
    eval_set = dataset[int(len(dataset) * 0.85):]
    train_set = dataset[: int(len(dataset) * 0.85)]

    if args.decode_sweep:
        run_decode_sweep(results)
    if args.ood_eval:
        run_ood_eval(eval_set, results)

    if args.hyperband:
        run_hyperband(eval_set, train_set, results)

    if args.train:
        only: Set[str] = set()
        for x in args.train:
            s = x.strip().upper()
            if not s.startswith("E"):
                s = "E" + s
            only.add(s)
        if "E33" in only:
            run_e33(eval_set, train_set, results)
        if "E34" in only:
            run_e34(eval_set, train_set, results)
        if "E35" in only:
            run_e35(eval_set, train_set, results)

    print(f"\nResults -> {RESULTS_PATH}")
    print("DONE.")


if __name__ == "__main__":
    main()
