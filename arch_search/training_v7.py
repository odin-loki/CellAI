"""Training harness: early-stop eval, best+last checkpoints, param freeze."""
from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from arch_search.harness import CheckpointTracker, OUT_DIR


class TrainHarness:
    """Extended reset-based training with periodic held-out eval."""

    def __init__(
        self,
        train_set: List[Tuple[str, int]],
        eval_fn: Callable[[nn.Module], Dict],
        headline_fn: Callable[[Dict], Optional[float]],
        segment_len: int = 64,
        log_every: int = 200,
        eval_every: int = 0,
        early_stop_patience: int = 0,
    ):
        self.train_set = train_set
        self.eval_fn = eval_fn
        self.headline_fn = headline_fn
        self.segment_len = segment_len
        self.log_every = log_every
        self.eval_every = eval_every
        self.early_stop_patience = early_stop_patience

    def train(
        self,
        model: nn.Module,
        n_steps: int,
        lr: float,
        exp_id: str,
        step_fn: Optional[Callable] = None,
    ) -> Dict:
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr * 0.05,
        )
        tracker = CheckpointTracker(model, exp_id)
        train_shuffled = list(self.train_set)
        random.shuffle(train_shuffled)
        data_iter = iter(train_shuffled)

        losses: List[float] = []
        steps: List[int] = []
        eval_log: List[Dict] = []
        run_loss = 0.0
        t0 = time.perf_counter()
        best_hl: Optional[float] = None
        stale = 0

        for step in range(1, n_steps + 1):
            if step == 1:
                print(
                    f"    … step 1/{n_steps} started (next log at {self.log_every})",
                    flush=True,
                )
            try:
                text, _ = next(data_iter)
            except StopIteration:
                random.shuffle(train_shuffled)
                data_iter = iter(train_shuffled)
                text, _ = next(data_iter)

            if step_fn:
                loss = step_fn(model, text, optimizer)
            else:
                loss = model.train_step_sequential(
                    text, optimizer, segment_len=self.segment_len, reset_state=True,
                )
            scheduler.step()
            run_loss += loss

            if step % self.log_every == 0:
                avg = run_loss / self.log_every
                lr_now = scheduler.get_last_lr()[0]
                tps = step / (time.perf_counter() - t0)
                print(
                    f"    step={step:>5}  loss={avg:.3f}  lr={lr_now:.2e}  {tps:.1f} s/s",
                    flush=True,
                )
                losses.append(avg)
                steps.append(step)
                run_loss = 0.0
                tracker.save(extra={"train": {"steps": steps, "losses": losses}}, tag="last")

            if self.eval_every and step % self.eval_every == 0:
                ppl = self.eval_fn(model)
                hl = self.headline_fn(ppl)
                eval_log.append({"step": step, "headline": hl, "ppl": ppl})
                print(f"    [eval @ {step}] text+code NLL={hl:.3f}", flush=True)
                if hl is not None:
                    tracker.save(
                        extra={"train": {"steps": steps, "losses": losses}, "ppl": ppl},
                        headline_value=hl,
                        tag="best",
                    )
                    if best_hl is None or hl < best_hl:
                        best_hl = hl
                        stale = 0
                    else:
                        stale += 1
                    if self.early_stop_patience and stale >= self.early_stop_patience:
                        print(f"    [early stop] no improve for {stale} evals", flush=True)
                        break

        return {
            "steps": steps,
            "losses": losses,
            "initial": losses[0] if losses else float("nan"),
            "final": losses[-1] if losses else float("nan"),
            "eval_log": eval_log,
            "best_headline": best_hl,
            "best_ckpt": str(tracker.best_path),
            "last_ckpt": str(tracker.last_path),
        }


def freeze_except(model: nn.Module, prefixes: Tuple[str, ...]) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(pref) for pref in prefixes)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
