"""
CellularAI v3 — Architecture Search Round 5 / Wave 2  (E27–E29)
================================================================
Implements roadmap Wave 1 (eval protocol + harness) and Wave 2 experiments:

  E27: Resume E24 (D=1024, 8k) → +8k fine-tune at LR=1e-4 (16k total)
  E28: Resume E22 (D=512, 16k) → +16k fine-tune at LR=5e-5 (32k total)
  E29: Resume E25 with fresh optimizer, LR=1e-4, EMA decay=0.999 (4k steps)

Headline metric: text+code avg NLL (protocol arch_eval_v1).

Run (from repo root):
    python -m arch_search.run_arch_search_v5 --reeval E22 E25
    python -m arch_search.run_arch_search_v5 --train E27
    python -m arch_search.run_arch_search_v5 --train E27 E28 E29
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arch_search.eval_protocol import EvalConfig, attach_protocol, load_eval_protocol
from arch_search.harness import RunMeta, OUT_DIR, save_results_atomic, write_meta_json
from arch_search.run_arch_search import ExperimentRunner, build_dataset
from arch_search.run_arch_search_v4 import (
    ContinuousCorpusTrainer,
    eval_ppl_continuous,
    gen_showcase,
    gen_showcase_warm,
    make_model,
)
from cellai_core.base import ModelParams
from v3.cell_ai_v3 import CellAIv3, V3Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_PATH = OUT_DIR / "results_v5.json"


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _champion_d512_cfg(label: str) -> Tuple[ModelParams, V3Config]:
    params = ModelParams(state_size=512, num_partitions=8)
    cfg = V3Config(
        label=label,
        pde_type="spectral",
        hebbian_type="sparse",
        partition_type="multiscale",
        resonance_type="none",
        N_fast=4,
        D_fast=256,
        N_slow=2,
        D_slow=512,
        K_slow=8,
        k_frac=0.125,
        gen_rep_penalty=1.3,
        gen_noise_std=0.03,
    )
    return params, cfg


def _champion_d1024_cfg(label: str) -> Tuple[ModelParams, V3Config]:
    params = ModelParams(state_size=1024, num_partitions=8)
    cfg = V3Config(
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
    return params, cfg


def build_model(exp: str) -> Tuple[CellAIv3, int]:
    if exp in ("E28", "E29"):
        params, cfg = _champion_d512_cfg(f"{exp.lower()}_d512")
    elif exp == "E27":
        params, cfg = _champion_d1024_cfg("E27_d1024_16k")
    else:
        raise ValueError(f"No model builder for {exp}")
    m = make_model(params, cfg)
    return m, sum(p.numel() for p in m.parameters())


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
    sd = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(sd["state_dict"])


# ─────────────────────────────────────────────────────────────────────────────
# EMA + continuous trainer (E29)
# ─────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential moving average of trainable weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone()

    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(p.data, alpha=1.0 - d)

    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])


class ContinuousTrainerEMA(ContinuousCorpusTrainer):
    """Continuous training with optional EMA and custom LR (E29)."""

    def __init__(
        self,
        texts: List[str],
        model: CellAIv3,
        shuffle_docs: bool = True,
        lr: float = 1e-4,
        ema_decay: Optional[float] = 0.999,
    ):
        super().__init__(texts, model, shuffle_docs=shuffle_docs)
        self.LR = lr
        self.ema = ModelEMA(model, ema_decay) if ema_decay else None


# Patch EMA update into ContinuousCorpusTrainer loop via subclass override
def _patch_continuous_ema_train() -> None:
    orig_train = ContinuousCorpusTrainer.train

    def train_with_ema(self, n_steps=4_000, save_path=None):
        if not isinstance(self, ContinuousTrainerEMA) or self.ema is None:
            return orig_train(self, n_steps=n_steps, save_path=save_path)

        import torch.nn.functional as F
        from arch_search.run_arch_search_v4 import DEVICE as DEV

        model = self.model
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.LR,
            weight_decay=1e-2,
        )
        warmup_steps = max(n_steps // 10, 50)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            prog = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
            return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        T = len(self.token_stream)
        pos = 0
        embs_all = model.encoder.embedding.weight
        scale = model.encoder._scale
        model.partitions.reset()
        model.memory_formation.reset()

        losses: List[float] = []
        steps: List[int] = []
        run_loss = 0.0
        t0 = time.perf_counter()

        for step in range(1, n_steps + 1):
            if pos + self.CHUNK_LEN + 1 >= T:
                pos = 0
                model.partitions.reset()
                model.memory_formation.reset()

            chunk = self.token_stream[pos : pos + self.CHUNK_LEN + 1]
            pos += self.CHUNK_LEN
            chunk_loss = 0.0
            chunk_count = 0
            optimizer.zero_grad()
            embs = embs_all[chunk] * scale

            for seg_start in range(0, self.CHUNK_LEN, self.SEG_LEN):
                seg_end = min(seg_start + self.SEG_LEN, self.CHUNK_LEN)
                seg_loss = torch.tensor(0.0, device=DEV)
                count = 0
                for t_local in range(seg_end - seg_start):
                    t = seg_start + t_local
                    inp = embs[t].detach()
                    st = model.cellular_step(inp)
                    logits = st @ embs_all.t()
                    nll = F.cross_entropy(logits.unsqueeze(0), chunk[t + 1].unsqueeze(0))
                    seg_loss = seg_loss + nll
                    count += 1
                if count > 0:
                    (seg_loss / count).backward()
                    chunk_loss += seg_loss.item()
                    chunk_count += count
                if hasattr(model.metaplasticity, "apply_pending_update"):
                    model.metaplasticity.apply_pending_update()
                if model._multiscale:
                    model.partitions.detach_state()
                else:
                    model.partitions._buffers["state"] = (
                        model.partitions._buffers["state"].detach()
                    )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            self.ema.update(model)

            avg_loss = chunk_loss / max(chunk_count, 1)
            run_loss += avg_loss
            if step % self.LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                tps = step / (time.perf_counter() - t0)
                avg = run_loss / self.LOG_EVERY
                print(
                    f"    step={step:>5}  loss={avg:.3f}  lr={lr:.2e}  "
                    f"pos={pos / len(self.token_stream) * 100:.1f}%  {tps:.1f} s/s",
                    flush=True,
                )
                losses.append(avg)
                steps.append(step)
                run_loss = 0.0

        result = {
            "steps": steps,
            "losses": losses,
            "initial": losses[0] if losses else float("nan"),
            "final": losses[-1] if losses else float("nan"),
        }
        if save_path:
            backup = self.ema.copy_to(model)
            torch.save(
                {"state_dict": model.state_dict(), "train": result, "ema_decay": self.ema.decay},
                save_path,
            )
            self.ema.restore(model, backup)
        return result

    ContinuousTrainerEMA.train = train_with_ema  # type: ignore[method-assign]


_patch_continuous_ema_train()


# ─────────────────────────────────────────────────────────────────────────────
# Eval wrapper
# ─────────────────────────────────────────────────────────────────────────────

def eval_with_protocol(
    model: CellAIv3,
    eval_set: List[Tuple[str, int]],
    cfg: EvalConfig,
) -> Dict:
    n = cfg.n_per_modality * 3
    warm = cfg.warm_tokens if cfg.regime == "continuous" else 0
    ppl = eval_ppl_continuous(
        model,
        eval_set,
        n=n,
        warm_tokens=warm,
        stream_burn_tokens=cfg.stream_burn_tokens,
    )
    return attach_protocol(ppl, cfg)


def headline_value(ppl: Dict) -> Optional[float]:
    v = ppl.get("text_code_avg_nll")
    if v is None or (isinstance(v, float) and v != v):
        mods = [ppl.get(m, {}).get("avg_nll") for m in ("text", "code")]
        mods = [x for x in mods if x is not None]
        return sum(mods) / len(mods) if mods else None
    return float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_e27(
    runner: ExperimentRunner,
    eval_set: List[Tuple[str, int]],
    train_texts: List[str],
    results: Dict,
) -> None:
    print("\n>>> E27: E24 RESUME +8k (D=1024, 16k total, LR=1e-4)")
    ckpt_in = OUT_DIR / "E24_D1024_8k.pt"
    if not ckpt_in.exists():
        print(f"  [SKIP] missing {ckpt_in}")
        return

    proto = load_eval_protocol("reset")
    model, n_params = build_model("E27")
    load_checkpoint(model, ckpt_in)
    print(f"  Parameters: {n_params:,}  (resumed from E24)")

    orig_steps, orig_lr = runner.TRAIN_STEPS, runner.LR
    runner.TRAIN_STEPS = 8_000
    runner.LR = 1e-4
    train_r = runner.train(model)
    runner.TRAIN_STEPS, runner.LR = orig_steps, orig_lr

    ppl = eval_with_protocol(model, eval_set, proto)
    hl = headline_value(ppl)
    print(f"  text+code NLL={hl:.3f}  macro PPL={ppl.get('macro_ppl', 0):.1f}")

    gen_r = gen_showcase(model, "E27")
    ckpt_out = OUT_DIR / "E27_D1024_16k.pt"
    torch.save({"state_dict": model.state_dict(), "train": train_r, "ppl": ppl}, ckpt_out)

    meta = RunMeta(
        experiment_id="E27",
        hypothesis="E24 resume +8k fine-tune; beat E22 text+code",
        protocol_id=proto.protocol_id,
        protocol_hash=proto.protocol_hash,
        regime="reset",
        train_steps=8000,
        lr=1e-4,
        resume_from=str(ckpt_in.name),
        headline_value=hl,
    )
    write_meta_json(meta, "E27_D1024_16k")

    results["E27"] = {
        "info": {"label": "E27_d1024_16k", "n_params": n_params, "resume_from": "E24"},
        "train": train_r,
        "ppl": ppl,
        "generation": gen_r,
        "meta": meta.__dict__,
    }
    save_results_atomic(RESULTS_PATH, results)
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_e28(
    runner: ExperimentRunner,
    eval_set: List[Tuple[str, int]],
    results: Dict,
) -> None:
    print("\n>>> E28: E22 RESUME +16k (D=512, 32k total, LR=5e-5)")
    ckpt_in = OUT_DIR / "E22_E20_16k.pt"
    if not ckpt_in.exists():
        print(f"  [SKIP] missing {ckpt_in}")
        return

    proto = load_eval_protocol("reset")
    model, n_params = build_model("E28")
    load_checkpoint(model, ckpt_in)

    orig_steps, orig_lr = runner.TRAIN_STEPS, runner.LR
    runner.TRAIN_STEPS = 16_000
    runner.LR = 5e-5
    train_r = runner.train(model)
    runner.TRAIN_STEPS, runner.LR = orig_steps, orig_lr

    ppl = eval_with_protocol(model, eval_set, proto)
    hl = headline_value(ppl)
    print(f"  text+code NLL={hl:.3f}  macro PPL={ppl.get('macro_ppl', 0):.1f}")

    gen_r = gen_showcase(model, "E28")
    ckpt_out = OUT_DIR / "E28_D512_32k.pt"
    torch.save({"state_dict": model.state_dict(), "train": train_r, "ppl": ppl}, ckpt_out)

    meta = RunMeta(
        experiment_id="E28",
        hypothesis="E22 resume +16k; beat E22 text+code at 32k total",
        protocol_id=proto.protocol_id,
        protocol_hash=proto.protocol_hash,
        regime="reset",
        train_steps=16000,
        lr=5e-5,
        resume_from=str(ckpt_in.name),
        headline_value=hl,
    )
    write_meta_json(meta, "E28_D512_32k")

    results["E28"] = {
        "info": {"label": "E28_d512_32k", "n_params": n_params, "resume_from": "E22"},
        "train": train_r,
        "ppl": ppl,
        "generation": gen_r,
        "meta": meta.__dict__,
    }
    save_results_atomic(RESULTS_PATH, results)
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def finalize_e29(
    eval_set: List[Tuple[str, int]],
    train_texts: List[str],
    results: Dict,
    do_generation: bool = True,
) -> None:
    """Eval + gen + save from existing E29 checkpoint (skip retrain)."""
    ckpt_out = OUT_DIR / "E29_ContinuousD512_EMA.pt"
    if not ckpt_out.exists():
        print(f"  [SKIP finalize] missing {ckpt_out}")
        return
    print("\n>>> E29 FINALIZE: load checkpoint, eval, gen, save")
    proto = load_eval_protocol("continuous")
    model, n_params = build_model("E29")
    ck = torch.load(str(ckpt_out), map_location=DEVICE)
    model.load_state_dict(ck["state_dict"])
    train_r = ck.get("train", {})

    ppl_cold = eval_ppl_continuous(model, eval_set, warm_tokens=0)
    ppl = eval_with_protocol(model, eval_set, proto)
    hl = headline_value(ppl)
    print(f"  text+code NLL (warm)={hl:.3f}  macro PPL={ppl.get('macro_ppl', 0):.1f}")

    gen_r = gen_showcase_warm(model, train_texts[:10]) if do_generation else []
    meta = RunMeta(
        experiment_id="E29",
        hypothesis="E25 fork with EMA + fresh optimizer; beat E25 warm text+code",
        protocol_id=proto.protocol_id,
        protocol_hash=proto.protocol_hash,
        regime="continuous",
        train_steps=4000,
        lr=1e-4,
        resume_from="E25_ContinuousD512.pt",
        headline_value=hl,
        extra={"ema_decay": 0.999, "finalized_from_checkpoint": True},
    )
    write_meta_json(meta, "E29_ContinuousD512_EMA")
    results["E29"] = {
        "info": {
            "label": "E29_continuous_d512_ema",
            "n_params": n_params,
            "resume_from": "E25",
            "ema_decay": 0.999,
        },
        "train": train_r,
        "ppl_cold": attach_protocol(ppl_cold, load_eval_protocol("reset")),
        "ppl": ppl,
        "generation": gen_r,
        "meta": meta.__dict__,
    }
    save_results_atomic(RESULTS_PATH, results)
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_e29(
    eval_set: List[Tuple[str, int]],
    train_texts: List[str],
    results: Dict,
) -> None:
    print("\n>>> E29: E25 FORK — fresh optimizer, LR=1e-4, EMA=0.999, 4k steps")
    ckpt_in = OUT_DIR / "E25_ContinuousD512.pt"
    if not ckpt_in.exists():
        print(f"  [SKIP] missing {ckpt_in}")
        return

    proto = load_eval_protocol("continuous")
    model, n_params = build_model("E29")
    load_checkpoint(model, ckpt_in)

    trainer = ContinuousTrainerEMA(
        train_texts, model, shuffle_docs=True, lr=1e-4, ema_decay=0.999
    )
    ckpt_out = OUT_DIR / "E29_ContinuousD512_EMA.pt"
    train_r = trainer.train(n_steps=4_000, save_path=ckpt_out)

    # Eval with EMA weights applied
    backup = trainer.ema.copy_to(model) if trainer.ema else {}
    ppl_cold = eval_ppl_continuous(model, eval_set, warm_tokens=0)
    ppl = eval_with_protocol(model, eval_set, proto)
    if trainer.ema:
        trainer.ema.restore(model, backup)

    hl = headline_value(ppl)
    print(f"  text+code NLL (warm)={hl:.3f}  macro PPL={ppl.get('macro_ppl', 0):.1f}")

    gen_r = gen_showcase_warm(model, train_texts[:10])

    meta = RunMeta(
        experiment_id="E29",
        hypothesis="E25 fork with EMA + fresh optimizer; beat E25 warm text+code",
        protocol_id=proto.protocol_id,
        protocol_hash=proto.protocol_hash,
        regime="continuous",
        train_steps=4000,
        lr=1e-4,
        resume_from=str(ckpt_in.name),
        headline_value=hl,
        extra={"ema_decay": 0.999},
    )
    write_meta_json(meta, "E29_ContinuousD512_EMA")

    results["E29"] = {
        "info": {
            "label": "E29_continuous_d512_ema",
            "n_params": n_params,
            "resume_from": "E25",
            "ema_decay": 0.999,
        },
        "train": train_r,
        "ppl_cold": attach_protocol(ppl_cold, load_eval_protocol("reset")),
        "ppl": ppl,
        "generation": gen_r,
        "meta": meta.__dict__,
    }
    save_results_atomic(RESULTS_PATH, results)
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def reeval_baselines(
    exp_ids: List[str],
    eval_set: List[Tuple[str, int]],
    train_texts: List[str],
    results: Dict,
    do_generation: bool = True,
) -> None:
    """Re-evaluate prior checkpoints under arch_eval_v1 protocol."""
    from arch_search.run_arch_search_v4 import _build_model_for_exp, _checkpoint_path

    for exp in exp_ids:
        exp = exp.strip().upper()
        if not exp.startswith("E"):
            exp = "E" + exp
        try:
            ck = _checkpoint_path(exp)
        except ValueError:
            print(f"  [SKIP reeval] unknown {exp}")
            continue
        if not ck.exists():
            print(f"  [SKIP reeval] missing {ck}")
            continue

        print(f"\n>>> --reeval {exp}: {ck.name}")
        model, n_params = _build_model_for_exp(exp)
        load_checkpoint(model, ck)

        if exp in ("E21", "E25", "E26", "E29"):
            proto = load_eval_protocol("continuous")
            print("  [PPL cold]")
            ppl_cold = eval_ppl_continuous(model, eval_set, warm_tokens=0)
            print("  [PPL stream-matched warm]")
            ppl = eval_with_protocol(model, eval_set, proto)
            gen = gen_showcase_warm(model, train_texts[:10]) if do_generation else []
            entry = {
                "info": {"label": exp, "n_params": n_params},
                "ppl_cold": attach_protocol(ppl_cold, load_eval_protocol("reset")),
                "ppl": ppl,
                "generation": gen,
                "reeval_protocol": proto.protocol_id,
            }
        else:
            proto = load_eval_protocol("reset")
            ppl = eval_with_protocol(model, eval_set, proto)
            gen = gen_showcase(model, exp) if do_generation else []
            entry = {
                "info": {"label": exp, "n_params": n_params},
                "ppl": ppl,
                "generation": gen,
                "reeval_protocol": proto.protocol_id,
            }

        hl = headline_value(ppl if "ppl" in entry else entry.get("ppl", {}))
        print(f"  text+code NLL={hl:.3f}" if hl else "  (no headline)")
        results[f"{exp}_reeval_v1"] = entry
        save_results_atomic(RESULTS_PATH, results)
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round 5 architecture search (E27–E29)")
    p.add_argument("--train", nargs="+", metavar="EXP", help="Run E27, E28, and/or E29")
    p.add_argument(
        "--reeval",
        nargs="+",
        metavar="EXP",
        help="Re-eval prior checkpoint under protocol v1 (e.g. E22 E25)",
    )
    p.add_argument("--no-gen", action="store_true", help="Skip generation during --reeval")
    p.add_argument(
        "--finalize",
        nargs="+",
        metavar="EXP",
        help="Eval+gen+save from existing checkpoint only (e.g. E29)",
    )
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
    print("CELL AI v3 — ARCH SEARCH ROUND 5 / WAVE 2 (E27–E29)")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    results: Dict = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            results = json.load(fh)

    print("\n--- Loading data ---")
    dataset = build_dataset(n_per=6_000)
    runner = ExperimentRunner(dataset)
    train_set = dataset[: int(len(dataset) * 0.85)]
    eval_set = dataset[int(len(dataset) * 0.85) :]
    train_texts = [t for t, _ in train_set]

    if args.finalize:
        for x in args.finalize:
            s = x.strip().upper()
            if not s.startswith("E"):
                s = "E" + s
            if s == "E29":
                finalize_e29(eval_set, train_texts, results, do_generation=not args.no_gen)
            else:
                print(f"  [SKIP finalize] no handler for {s}")
        print("\nDONE (--finalize).")
        return

    if args.reeval:
        reeval_baselines(args.reeval, eval_set, train_texts, results, do_generation=not args.no_gen)
        print("\nDONE (--reeval).")
        return

    if not args.train:
        print("\nNo --train or --reeval. See arch_search/experiments.yaml")
        print("  python -m arch_search.run_arch_search_v5 --reeval E22 E25")
        print("  python -m arch_search.run_arch_search_v5 --train E27")
        return

    only: Set[str] = set()
    for x in args.train:
        s = x.strip().upper()
        if not s.startswith("E"):
            s = "E" + s
        only.add(s)

    if "E27" in only:
        run_e27(runner, eval_set, train_texts, results)
    if "E28" in only:
        run_e28(runner, eval_set, results)
    if "E29" in only:
        run_e29(eval_set, train_texts, results)

    print("\n" + "=" * 72)
    print("WAVE 2 SUMMARY")
    print("=" * 72)
    print(f"{'Exp':<6} {'text+code NLL':>14} {'macro PPL':>12}")
    print("-" * 36)
    for exp in ("E27", "E28", "E29"):
        r = results.get(exp, {})
        ppl = r.get("ppl", {})
        tc = ppl.get("text_code_avg_nll", "—")
        mp = ppl.get("macro_ppl", "—")
        if isinstance(tc, float):
            tc = f"{tc:.3f}"
        if isinstance(mp, float):
            mp = f"{mp:.1f}"
        print(f"{exp:<6} {str(tc):>14} {str(mp):>12}")
    print(f"\nResults -> {RESULTS_PATH}")
    print("DONE.")


if __name__ == "__main__":
    main()
