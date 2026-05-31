"""
CellularAI v3 — Architecture Search Round 4  (E21–E24)
======================================================
Builds on the Round 3 champion (E20: SpectralPDE+MultiScale+SparseHebbian, D=512, 8k steps)
and explores three remaining growth axes:

  E21: Continuous-context training (D=256)
       Concatenate corpus → 256-token chunks → NO state reset between chunks.
       This extends the effective context from 64 tokens (BPTT segment) to the
       full corpus length, enabling long-range dependency learning.

  E22: E20 extended to 16k steps (resumed from checkpoint)
       Validates that the D=512 champion continues improving past 8k.

  E23: D=1024 scaling test — SpectralPDE+MultiScale+SparseHebbian, 4k steps
       Extrapolates the D=256→512 PPL trend (246.6) to D=1024.

  E24: D=1024, 8k steps — final champion candidate
       Full-scale training to establish the performance ceiling.

  E25: D=512 continuous training with shuffle_docs=True (domain-interleaved stream)

New training infrastructure:
  - ContinuousCorpusDataset: tokenises the full corpus into a flat token stream,
    yields fixed-length chunks without state reset between chunks.
  - train_continuous(): trains using the corpus stream, carries state across
    chunk boundaries (truncated BPTT every SEG_LEN=64 tokens still, but no
    full reset until epoch boundary).

Results saved to data/local/arch_search/results_v4.json after each experiment
(incremental write via .tmp + replace) so crashes after E23 still keep E21–E23.

Run (from repo root; unbuffered when piping to a file):
    .venv\\Scripts\\python -u -m arch_search.run_arch_search_v4 2>&1 | Tee-Object C:\\Temp\\arch_v4.log

CLI (optional):
    python -m arch_search.run_arch_search_v4 --reeval E25              # refresh PPL (+ gen) from checkpoint
    python -m arch_search.run_arch_search_v4 --reeval E21 E25 --no-gen
    python -m arch_search.run_arch_search_v4 --train E21             # force E21 training + eval (overwrite JSON)
    python -m arch_search.run_arch_search_v4 --train E26             # E25 checkpoint + 4k more continuous steps (8k total)
    python -m arch_search.run_arch_search_v4 --ablation-burn         # E25: NLL vs burn 2048/4096/8192 (n=30), then exit

Batch (ablation, then E21, then E26 if E25 ckpt exists):
    python -m arch_search.run_round4_followup
    # or:  .\\arch_search\\run_round4_followup.ps1
"""
from __future__ import annotations

import argparse
import gc, json, math, os, random, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

MODALITY_NAMES = ["text", "code", "math"]

random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True

print("=" * 72)
print("CELL AI v3 — ARCH SEARCH ROUND 4 (E21-E26)")
print("=" * 72)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from arch_search.run_arch_search import build_dataset, ExperimentRunner, MODALITY_NAMES as MN, OUT_DIR as _OD
from cellai_core.base import ModelParams
from v3.cell_ai_v3   import CellAIv3, V3Config


# ─────────────────────────────────────────────────────────────────────────────
# Continuous corpus dataset
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousCorpusTrainer:
    """
    Trains on the entire corpus as a single flat token stream.

    Instead of resetting cellular state between texts, the model carries
    state across document boundaries. State is ONLY detached (not zeroed)
    at each BPTT segment boundary (every SEG_LEN tokens) — this gives
    the model an effective context window spanning the entire corpus.

    Training procedure per step:
      1. Take next CHUNK_LEN tokens from the stream.
      2. Process tokens 0..CHUNK_LEN-1, accumulate NTP loss.
      3. Backward every SEG_LEN tokens (truncated BPTT).
      4. Apply optimizer step.
      5. State carries over to the next chunk.
      6. At epoch boundary: reset state and shuffle corpus order.

    This mirrors the standard GPT pretraining setup, adapted for the
    cellular state-space formulation.
    """

    CHUNK_LEN = 256    # tokens processed per training step
    SEG_LEN   = 64     # BPTT truncation length (same as before)
    LR        = 3e-4
    LOG_EVERY = 100

    def __init__(
        self,
        texts:        List[str],
        model:        CellAIv3,
        sep_token:    int  = 100_257,   # cl100k_base <|endoftext|>
        shuffle_docs: bool = False,      # shuffle before concat to mix domains
    ):
        self.model     = model
        self.sep_token = sep_token

        # Optionally shuffle documents to interleave domains
        if shuffle_docs:
            texts = list(texts)
            random.shuffle(texts)
            print("  [Shuffled documents (domain-interleaved)]")

        # Build full token stream once
        print("  [Tokenising corpus for continuous training...]", flush=True)
        all_toks: List[int] = []
        for text in texts:
            t = model.encoder.tokenize(text)
            if t:
                all_toks.extend(t)
                all_toks.append(sep_token)
        self.token_stream = torch.tensor(all_toks, dtype=torch.long, device=DEVICE)
        print(f"  Corpus tokens: {len(all_toks):,}  ({len(texts):,} docs)")

    def train(
        self,
        n_steps:   int = 4_000,
        save_path: Optional[Path] = None,
    ) -> Dict:
        model    = self.model
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.LR, weight_decay=1e-2,
        )
        # Warmup for first 10% of steps, then cosine decay
        warmup_steps = max(n_steps // 10, 50)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            prog = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
            return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        T        = len(self.token_stream)
        pos      = 0        # current position in token stream
        embs_all = model.encoder.embedding.weight  # (V, D)
        scale    = model.encoder._scale

        # Reset state at epoch start
        model.partitions.reset()
        model.memory_formation.reset()

        losses: List[float] = []
        steps:  List[int]   = []
        run_loss = 0.0
        t0 = time.perf_counter()

        for step in range(1, n_steps + 1):
            # Wrap around at corpus end (new epoch: reset state)
            if pos + self.CHUNK_LEN + 1 >= T:
                pos = 0
                model.partitions.reset()
                model.memory_formation.reset()

            # Slice chunk tokens
            chunk = self.token_stream[pos : pos + self.CHUNK_LEN + 1]  # +1 for targets
            pos  += self.CHUNK_LEN

            chunk_loss  = 0.0
            chunk_count = 0
            optimizer.zero_grad()

            # Slice embeddings for the chunk
            embs = embs_all[chunk] * scale  # (CHUNK_LEN+1, D)

            # BPTT within chunk
            for seg_start in range(0, self.CHUNK_LEN, self.SEG_LEN):
                seg_end  = min(seg_start + self.SEG_LEN, self.CHUNK_LEN)
                seg_loss = torch.tensor(0.0, device=DEVICE)
                count    = 0

                for t_local in range(seg_end - seg_start):
                    t    = seg_start + t_local
                    inp  = embs[t].detach()  # detach embedding lookup from outer graph
                    st   = model.cellular_step(inp)
                    logits = st @ embs_all.t()
                    nll    = F.cross_entropy(logits.unsqueeze(0), chunk[t+1].unsqueeze(0))
                    seg_loss = seg_loss + nll
                    count   += 1

                if count > 0:
                    (seg_loss / count).backward()
                    chunk_loss  += seg_loss.item()
                    chunk_count += count

                # Apply deferred Hebbian AFTER backward
                if hasattr(model.metaplasticity, 'apply_pending_update'):
                    model.metaplasticity.apply_pending_update()

                # Carry state (detach for next BPTT segment)
                if model._multiscale:
                    model.partitions.detach_state()
                else:
                    model.partitions._buffers["state"] = \
                        model.partitions._buffers["state"].detach()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            avg_loss = chunk_loss / max(chunk_count, 1)
            run_loss += avg_loss

            if step % self.LOG_EVERY == 0:
                lr  = scheduler.get_last_lr()[0]
                tps = step / (time.perf_counter() - t0)
                avg = run_loss / self.LOG_EVERY
                print(f"    step={step:>5}  loss={avg:.3f}  lr={lr:.2e}  "
                      f"pos={pos/len(self.token_stream)*100:.1f}%  {tps:.1f} s/s", flush=True)
                losses.append(avg)
                steps.append(step)
                run_loss = 0.0

        if save_path:
            torch.save({"state_dict": model.state_dict(),
                        "train": {"steps": steps, "losses": losses}}, save_path)
        return {"steps": steps, "losses": losses,
                "initial": losses[0] if losses else float("nan"),
                "final":   losses[-1] if losses else float("nan")}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_model(params: ModelParams, cfg: V3Config) -> CellAIv3:
    return CellAIv3(params, cfg).to(DEVICE)


# Match ContinuousCorpusTrainer.SEG_LEN — partition detach must happen on the same cadence
# as truncated BPTT during training, or recurrent state drifts to an OOD attractor.
_CONTINUOUS_BPTT_SEG: int = 64
# Tokens to process before measuring NLL so memory/partitions match training steady-state.
_CONTINUOUS_STREAM_BURN: int = 4096


def _detach_partition_state(model: CellAIv3) -> None:
    if model._multiscale:
        model.partitions.detach_state()
    else:
        st = model.partitions._buffers["state"]
        model.partitions._buffers["state"] = st.detach()


def _gather_token_stream(
    encoder,
    corpus_texts: List[str],
    start_doc_i: int,
    min_tokens: int,
    sep_token: int = 100_257,
) -> List[int]:
    """Concatenate tokenised docs (with sep) until at least `min_tokens` ids."""
    out: List[int] = []
    n = len(corpus_texts)
    if n == 0 or min_tokens <= 0:
        return out
    d = start_doc_i % n
    scanned = 0
    while len(out) < min_tokens and scanned < n + 4:
        toks = encoder.tokenize(corpus_texts[d])
        if toks:
            out.extend(toks)
            out.append(sep_token)
        d = (d + 1) % n
        scanned += 1
    return out[:min_tokens]


def eval_ppl_continuous(model: CellAIv3, eval_set: List[Tuple[str,int]],
                        n: int = 150,
                        warm_tokens: int = 0,
                        stream_burn_tokens: Optional[int] = None) -> Dict:
    """
    PPL eval supporting two modes:

    warm_tokens=0 (reset-based, default):
        Standard per-sample evaluation — reset state before each sample.
        Use for models trained with reset_state=True (E0-E20).

    warm_tokens>0 (continuous / stream-trained models):
        For each eval text, reset once, then run a long *stream burn-in* (default ≥4096 tokens
        from rotating held-out docs), with partition detach every 64 tokens — matching
        ContinuousCorpusTrainer BPTT.  Short warm prefixes without detach were invalid:
        recurrent state diverged from the training trajectory (spurious 10³–10⁴ NLL).

        The `warm_tokens` argument is kept for API compatibility; the effective burn length
        is ``stream_burn_tokens`` if set, else ``max(warm_tokens, _CONTINUOUS_STREAM_BURN)``.
    """
    model.eval()
    W_snap = model.metaplasticity.W.data.clone()
    per_mod: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    samples = {label: [t for t, l in eval_set if l == label][:n//3] for label in range(3)}

    all_texts = [t for t, _ in eval_set]
    embs_w = model.encoder.embedding.weight
    scale  = model.encoder._scale

    for label, texts in samples.items():
        for i, text in enumerate(texts):
            model.metaplasticity.W.data.copy_(W_snap)
            model.partitions.reset()
            model.memory_formation.reset()

            toks = model.encoder.tokenize(text)[:256]
            if len(toks) < 2:
                continue
            eval_ids = torch.tensor(toks, dtype=torch.long, device=DEVICE)

            if warm_tokens > 0:
                burn_n = (
                    stream_burn_tokens
                    if stream_burn_tokens is not None
                    else max(warm_tokens, _CONTINUOUS_STREAM_BURN)
                )
                start_doc = (i + len(texts)) % max(len(all_texts), 1)
                burn_list = _gather_token_stream(
                    model.encoder, all_texts, start_doc, burn_n,
                )
                if len(burn_list) < burn_n:
                    continue
                burn_ids = torch.tensor(burn_list, dtype=torch.long, device=DEVICE)
                full_ids = torch.cat([burn_ids, eval_ids])
                embs = model.encoder.embedding(full_ids) * scale
                B = burn_ids.shape[0]
                L = eval_ids.shape[0]
                nll_total, count = 0.0, 0
                t = 0
                last_i = B + L - 2
                with torch.no_grad():
                    while t <= last_i:
                        seg_end = min(t + _CONTINUOUS_BPTT_SEG, last_i + 1)
                        for j in range(t, seg_end):
                            state = model.cellular_step(embs[j])
                            if B - 1 <= j < B + L - 1:
                                logits = state @ embs_w.t()
                                nll = F.cross_entropy(
                                    logits.unsqueeze(0), full_ids[j + 1].unsqueeze(0)
                                )
                                nll_total += nll.item()
                                count += 1
                        _detach_partition_state(model)
                        t = seg_end
                if count > 0:
                    per_mod[label].append(nll_total / count)
            else:
                tok_ids = eval_ids
                embs = model.encoder.embedding(tok_ids) * scale
                nll_total, count = 0.0, 0
                for t in range(len(toks) - 1):
                    with torch.no_grad():
                        state = model.cellular_step(embs[t])
                        logits = state @ embs_w.t()
                        nll = F.cross_entropy(
                            logits.unsqueeze(0), tok_ids[t + 1].unsqueeze(0)
                        )
                        nll_total += nll.item()
                        count += 1
                if count > 0:
                    per_mod[label].append(nll_total / count)

    model.metaplasticity.W.data.copy_(W_snap)
    results = {}
    for label, nlls in per_mod.items():
        if nlls:
            avg = sum(nlls) / len(nlls)
            results[MODALITY_NAMES[label]] = {"avg_nll": avg, "ppl": math.exp(min(avg, 30.0))}
    all_nll = [v["avg_nll"] for v in results.values()]
    results["macro_avg_nll"] = sum(all_nll) / len(all_nll) if all_nll else float("nan")
    results["macro_ppl"]     = math.exp(min(results["macro_avg_nll"], 30.0))
    return results


def _safe_console_snippet(s: str, max_len: int = 80) -> str:
    """Windows cp1252 consoles crash on e.g. ≤ (U+2264); never let print() raise."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return s[:max_len].encode(enc, errors="replace").decode(enc, errors="replace")


def _run_token_ids_segmented(model: CellAIv3, ids: torch.Tensor, scale: float) -> None:
    """Forward through token ids with BPTT detach cadence (no loss)."""
    if ids.numel() == 0:
        return
    embs = model.encoder.embedding(ids) * scale
    t, T = 0, int(ids.shape[0])
    with torch.no_grad():
        while t < T:
            seg_end = min(t + _CONTINUOUS_BPTT_SEG, T)
            for j in range(t, seg_end):
                model.cellular_step(embs[j])
            _detach_partition_state(model)
            t = seg_end


def gen_showcase_warm(model: CellAIv3, warm_texts: List[str],
                     n_prompts: int = 6) -> List[Dict]:
    """
    Generation with warm cellular state.
    Burns a long token stream (same cadence as continuous PPL eval) then calls
    generate(reset_state=False) so the warm-up is not discarded.
    """
    prompts = [
        ("math",  "Solve for x: 3x^2 - 12 = 0"),
        ("math",  "The integral of sin(x) from 0 to pi is"),
        ("code",  "def binary_search(arr, target):"),
        ("code",  "class Stack:\n    def __init__(self):"),
        ("text",  "The cellular automaton model predicts"),
        ("text",  "Deep learning achieves state of the art"),
    ]
    results = []
    W_snap  = model.metaplasticity.W.data.clone()
    scale   = model.encoder._scale

    for i, (domain, prompt) in enumerate(prompts[:n_prompts]):
        model.metaplasticity.W.data.copy_(W_snap)
        model.partitions.reset()
        model.memory_formation.reset()

        burn_list = _gather_token_stream(
            model.encoder, warm_texts, i % max(len(warm_texts), 1), _CONTINUOUS_STREAM_BURN,
        )
        if burn_list:
            w_ids = torch.tensor(burn_list, dtype=torch.long, device=DEVICE)
            _run_token_ids_segmented(model, w_ids, scale)

        gen  = model.generate(prompt, max_tokens=48, temperature=0.85,
                              rep_penalty=1.3, noise_std=0.03, reset_state=False)
        cont = gen[len(prompt):]
        toks = model.encoder.tokenize(cont)
        div  = len(set(toks)) / max(len(toks), 1)
        results.append({"domain": domain, "prompt": prompt,
                        "continuation": cont, "diversity": div, "warmed": True})
        print(f"    [{domain}] '{_safe_console_snippet(prompt, 40)}'")
        print(f"           -> '{_safe_console_snippet(cont, 60)}'  [div={div:.2f}]", flush=True)

    model.metaplasticity.W.data.copy_(W_snap)
    return results


def gen_showcase(model: CellAIv3, label: str) -> List[Dict]:
    prompts = [
        ("math",  "Solve for x: 3x^2 - 12 = 0"),
        ("math",  "The integral of sin(x) from 0 to pi is"),
        ("code",  "def binary_search(arr, target):"),
        ("code",  "class Stack:\n    def __init__(self):"),
        ("text",  "The cellular automaton model predicts"),
        ("text",  "Deep learning achieves state of the art"),
    ]
    results = []
    W_snap = model.metaplasticity.W.data.clone()
    for domain, prompt in prompts:
        model.metaplasticity.W.data.copy_(W_snap)
        gen  = model.generate(prompt, max_tokens=48, temperature=0.85,
                              rep_penalty=1.3, noise_std=0.03)
        cont = gen[len(prompt):]
        toks = model.encoder.tokenize(cont)
        div  = len(set(toks)) / max(len(toks), 1)
        results.append({"domain": domain, "prompt": prompt,
                        "continuation": cont, "diversity": div})
        print(f"    [{domain}] '{_safe_console_snippet(prompt, 40)}'")
        print(f"           -> '{_safe_console_snippet(cont, 60)}'  [div={div:.2f}]", flush=True)
    model.metaplasticity.W.data.copy_(W_snap)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _save_v4_results(results: Dict) -> None:
    """Atomic-ish write so interrupted runs keep E21–E24 progress."""
    path = OUT_DIR / "results_v4.json"
    tmp  = OUT_DIR / "results_v4.json.tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(path)
    print(f"  [saved] {path}  ({len(results)} experiments)")


def _want_exp(exp: str, results: Dict, only_train: Optional[Set[str]]) -> bool:
    """If only_train is set, run only listed experiments; else run if missing from results."""
    if only_train is not None:
        return exp in only_train
    return exp not in results


def _want_e26(only_train: Optional[Set[str]]) -> bool:
    """E26 never auto-runs on a full pipeline pass; use --train E26."""
    return only_train is not None and "E26" in only_train


def _build_model_for_exp(exp: str) -> Tuple[CellAIv3, int]:
    """Construct fresh CellAIv3 for Round-4 experiment id (E21–E26 family)."""
    if exp == "E21":
        params = ModelParams(state_size=256, num_partitions=4)
        cfg = V3Config(label="E21_continuous_d256",
                       pde_type="spectral", hebbian_type="sparse",
                       partition_type="multiscale", resonance_type="none",
                       N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8,
                       k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m = make_model(params, cfg)
        return m, sum(p.numel() for p in m.parameters())
    if exp == "E22":
        params = ModelParams(state_size=512, num_partitions=8)
        cfg = V3Config(label="E22_e20_16k",
                       pde_type="spectral", hebbian_type="sparse",
                       partition_type="multiscale", resonance_type="none",
                       N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                       k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m = make_model(params, cfg)
        return m, sum(p.numel() for p in m.parameters())
    if exp in ("E23", "E24"):
        params = ModelParams(state_size=1024, num_partitions=8)
        lab = "E23_d1024_4k" if exp == "E23" else "E24_d1024_8k"
        cfg = V3Config(label=lab,
                       pde_type="spectral", hebbian_type="sparse",
                       partition_type="multiscale", resonance_type="none",
                       N_fast=8, D_fast=256, N_slow=4, D_slow=512, K_slow=8,
                       k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m = make_model(params, cfg)
        return m, sum(p.numel() for p in m.parameters())
    if exp in ("E25", "E26"):
        params = ModelParams(state_size=512, num_partitions=8)
        cfg = V3Config(label="E25_continuous_d512",
                       pde_type="spectral", hebbian_type="sparse",
                       partition_type="multiscale", resonance_type="none",
                       N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                       k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m = make_model(params, cfg)
        return m, sum(p.numel() for p in m.parameters())
    raise ValueError(f"Unknown experiment for model build: {exp}")


def _checkpoint_path(exp: str) -> Path:
    m = {
        "E21": "E21_ContinuousD256.pt",
        "E22": "E22_E20_16k.pt",
        "E23": "E23_D1024_4k.pt",
        "E24": "E24_D1024_8k.pt",
        "E25": "E25_ContinuousD512.pt",
        "E26": "E26_ContinuousD512_8k.pt",
    }
    if exp not in m:
        raise ValueError(f"No checkpoint mapping for {exp}")
    return OUT_DIR / m[exp]


def reeval_round4_experiments(
    exp_ids: List[str],
    results: Dict,
    eval_set: List[Tuple[str, int]],
    train_texts: List[str],
    runner: ExperimentRunner,
    do_generation: bool = True,
) -> None:
    """Load each checkpoint, refresh PPL (and optional gen), merge into ``results``."""
    for exp in exp_ids:
        exp = exp.strip().upper()
        if not exp.startswith("E"):
            exp = "E" + exp
        ck = _checkpoint_path(exp)
        if not ck.exists():
            print(f"  [SKIP {exp}] missing checkpoint {ck}")
            continue
        print(f"\n>>> --reeval {exp}: loading {ck.name}")
        m, n_params = _build_model_for_exp(exp)
        sd = torch.load(str(ck), map_location=DEVICE)
        m.load_state_dict(sd["state_dict"])
        entry: Dict = dict(results.get(exp, {}))
        entry["info"] = {**(entry.get("info") or {}), "n_params": n_params}

        if exp in ("E21", "E25", "E26"):
            print("  [PPL cold]")
            entry["ppl_cold"] = eval_ppl_continuous(m, eval_set, warm_tokens=0)
            print("  [PPL stream-matched warm]")
            entry["ppl"] = eval_ppl_continuous(m, eval_set, warm_tokens=128)
            if do_generation:
                print("  [gen_showcase_warm]")
                entry["generation"] = gen_showcase_warm(m, train_texts[:10])
        else:
            print("  [PPL reset-based]")
            entry["ppl"] = eval_ppl_continuous(m, eval_set, warm_tokens=0)
            if do_generation:
                print("  [gen_showcase]")
                entry["generation"] = gen_showcase(m, exp)
        entry["reeval_note"] = time.strftime("%Y-%m-%dT%H:%M:%S") + " via --reeval"
        results[exp] = entry
        del m
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        _save_v4_results(results)
        print(f"  [done] {exp}")


def run_stream_burn_ablation(
    eval_set: List[Tuple[str, int]],
    n_samples: int = 30,
) -> None:
    """Print macro NLL for several burn lengths (E25 checkpoint)."""
    ck = OUT_DIR / "E25_ContinuousD512.pt"
    if not ck.exists():
        print(f"[ablation-burn] missing {ck}")
        return
    m, _ = _build_model_for_exp("E25")
    m.load_state_dict(torch.load(str(ck), map_location=DEVICE)["state_dict"])
    print("\n>>> Stream burn ablation (E25 checkpoint, n=%d per modality)" % (n_samples // 3 * 3))
    print(f"{'burn_tokens':>12}  {'macro_nll':>10}  {'ppl@cap20':>12}  (exp(min(nll,20)) for display)")
    for burn in (2048, 4096, 8192):
        r = eval_ppl_continuous(
            m, eval_set, n=n_samples, warm_tokens=128, stream_burn_tokens=burn,
        )
        macro = r.get("macro_avg_nll", float("nan"))
        ppl_c = math.exp(min(macro, 20.0)) if macro == macro else float("nan")
        print(f"{burn:>12}  {macro:>10.3f}  {ppl_c:>12.1f}")
    del m
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Round 4 architecture search (E21–E26) and checkpoint re-evaluation.",
    )
    p.add_argument(
        "--reeval",
        nargs="+",
        metavar="EXP",
        help="Reload checkpoint(s) and refresh PPL in results_v4.json (e.g. E21 E25)",
    )
    p.add_argument(
        "--train",
        nargs="+",
        metavar="EXP",
        help="Force-run only/for listed training blocks (E21, E26, …). E26 = resume E25 + 4k steps.",
    )
    p.add_argument(
        "--no-gen",
        action="store_true",
        help="With --reeval, skip generation showcase",
    )
    p.add_argument(
        "--ablation-burn",
        action="store_true",
        help="Print macro NLL vs stream burn 2048/4096/8192 using E25 checkpoint; then exit",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
        except Exception:
            try:
                sys.stdout.reconfigure(line_buffering=True)
                sys.stderr.reconfigure(line_buffering=True)
            except Exception:
                pass

    results: Dict = {}
    if (OUT_DIR / "results_v4.json").exists():
        with open(OUT_DIR / "results_v4.json") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    print("\n--- Loading data ---")
    dataset = build_dataset(n_per=6_000)
    runner  = ExperimentRunner(dataset)

    train_set  = dataset[:int(len(dataset) * 0.85)]
    eval_set   = dataset[int(len(dataset) * 0.85):]
    train_texts = [t for t, _ in train_set]

    if args.ablation_burn:
        run_stream_burn_ablation(eval_set)
        print("DONE (--ablation-burn).")
        return

    if args.reeval:
        reeval_norm = []
        for x in args.reeval:
            s = x.strip().upper()
            if not s.startswith("E"):
                s = "E" + s
            reeval_norm.append(s)
        reeval_round4_experiments(
            reeval_norm, results, eval_set, train_texts, runner,
            do_generation=not args.no_gen,
        )
        print("DONE (--reeval).")
        return

    only_train: Optional[Set[str]] = None
    if args.train:
        only_train = set()
        for x in args.train:
            s = x.strip().upper()
            if not s.startswith("E"):
                s = "E" + s
            only_train.add(s)

    # ── E21: Continuous-context training at D=256 ─────────────────────────────
    if _want_exp("E21", results, only_train):
        print("\n\n>>> E21: CONTINUOUS-CONTEXT TRAINING (D=256, no state reset, 4k steps)")
        params = ModelParams(state_size=256, num_partitions=4)
        cfg    = V3Config(label="E21_continuous_d256",
                          pde_type="spectral", hebbian_type="sparse",
                          partition_type="multiscale", resonance_type="none",
                          N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8,
                          k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m21 = make_model(params, cfg)
        n21 = sum(p.numel() for p in m21.parameters())
        print(f"  Parameters: {n21:,}")

        trainer21 = ContinuousCorpusTrainer(train_texts, m21)
        train_r   = trainer21.train(n_steps=4_000, save_path=OUT_DIR/"E21_ContinuousD256.pt")

        print("\n  [Perplexity eval - cold state (reset-based, for comparison)]")
        ppl_cold = eval_ppl_continuous(m21, eval_set, warm_tokens=0)
        print(f"    Macro NLL (cold) = {ppl_cold.get('macro_avg_nll', 0):.3f}")

        print("\n  [Perplexity eval - stream-matched warm (burn≥4096, BPTT detach/64)]")
        ppl_r = eval_ppl_continuous(m21, eval_set, warm_tokens=128)
        for mod, stats in ppl_r.items():
            if isinstance(stats, dict):
                print(f"    {mod:<8}  nll={stats['avg_nll']:.3f}  ppl={stats['ppl']:,.0f}")
        print(f"    Macro NLL = {ppl_r.get('macro_avg_nll', 0):.3f}")
        print(f"    Macro PPL = {ppl_r.get('macro_ppl', 0):.1f}")

        print("\n  [Throughput]")
        tput_r = runner.profile_throughput(m21)
        print(f"    Full forward: {tput_r['full_ms']:.3f} ms  ({tput_r['calls_per_s']:.0f}/s)")

        print("\n  [Generation (with warm-up prefix)]")
        gen_r = gen_showcase_warm(m21, train_texts[:10])

        results["E21"] = {
            "info":       {"label": "E21_continuous_d256", "n_params": n21},
            "train":      train_r,
            "ppl_cold":   ppl_cold,
            "ppl":        ppl_r,
            "throughput": tput_r,
            "generation": gen_r,
        }
        del m21; gc.collect(); torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
        _save_v4_results(results)

    # ── E22: Resume E20 checkpoint → 16k total steps ─────────────────────────
    if _want_exp("E22", results, only_train):
        print("\n\n>>> E22: E20 EXTENDED TO 16k STEPS (resume checkpoint)")
        ckpt_path = OUT_DIR / "E20_Champion_SparseD512_8k.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        else:
            params = ModelParams(state_size=512, num_partitions=8)
            cfg    = V3Config(label="E22_e20_16k",
                              pde_type="spectral", hebbian_type="sparse",
                              partition_type="multiscale", resonance_type="none",
                              N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                              k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
            m22 = make_model(params, cfg)
            ck  = torch.load(str(ckpt_path), map_location=DEVICE)
            m22.load_state_dict(ck["state_dict"])
            n22 = sum(p.numel() for p in m22.parameters())
            print(f"  Parameters: {n22:,}  (resumed from 8k checkpoint)")

            # Train for another 8k steps with fresh cosine schedule (reduced LR for fine-tune)
            orig_steps, orig_lr = runner.TRAIN_STEPS, runner.LR
            runner.TRAIN_STEPS = 8_000
            runner.LR          = 1e-4
            train_r22 = runner.train(m22)
            runner.TRAIN_STEPS, runner.LR = orig_steps, orig_lr

            print("\n  [Perplexity eval]")
            ppl_r22 = eval_ppl_continuous(m22, eval_set)
            for mod, stats in ppl_r22.items():
                if isinstance(stats, dict):
                    print(f"    {mod:<8}  nll={stats['avg_nll']:.3f}  ppl={stats['ppl']:,.0f}")
            print(f"    Macro NLL = {ppl_r22.get('macro_avg_nll',0):.3f}  PPL = {ppl_r22.get('macro_ppl',0):.1f}")

            print("\n  [Generation]")
            gen_r22 = gen_showcase(m22, "E22")

            torch.save({"state_dict": m22.state_dict(),
                        "result": {"train": train_r22, "ppl": ppl_r22}},
                       OUT_DIR/"E22_E20_16k.pt")
            results["E22"] = {
                "info":       {"label": "E22_e20_16k", "n_params": n22},
                "train":      train_r22,
                "ppl":        ppl_r22,
                "generation": gen_r22,
            }
            del m22; gc.collect(); torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
            _save_v4_results(results)

    # ── E23: D=1024 scaling, 4k steps ─────────────────────────────────────────
    if _want_exp("E23", results, only_train):
        print("\n\n>>> E23: D=1024 SCALING TEST (SpectralPDE+MultiScale+SparseHebbian, 4k steps)")
        params = ModelParams(state_size=1024, num_partitions=8)
        cfg    = V3Config(label="E23_d1024_4k",
                          pde_type="spectral", hebbian_type="sparse",
                          partition_type="multiscale", resonance_type="none",
                          N_fast=8, D_fast=256, N_slow=4, D_slow=512, K_slow=8,
                          k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m23 = make_model(params, cfg)
        n23 = sum(p.numel() for p in m23.parameters())
        print(f"  Parameters: {n23:,}")
        results["E23"] = runner.run_experiment(m23, "E23_D1024_4k", n_steps=4_000)
        _save_v4_results(results)

        print("\n  [E23 generation]")
        gen_r23 = gen_showcase(m23, "E23")
        results["E23"]["generation_ext"] = gen_r23
        del m23; gc.collect(); torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
        _save_v4_results(results)

    # ── E24: D=1024 champion, 8k steps ────────────────────────────────────────
    if _want_exp("E24", results, only_train):
        print("\n\n>>> E24: D=1024 CHAMPION (8k steps — final scale target)")
        params = ModelParams(state_size=1024, num_partitions=8)
        cfg    = V3Config(label="E24_d1024_8k",
                          pde_type="spectral", hebbian_type="sparse",
                          partition_type="multiscale", resonance_type="none",
                          N_fast=8, D_fast=256, N_slow=4, D_slow=512, K_slow=8,
                          k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m24 = make_model(params, cfg)
        n24 = sum(p.numel() for p in m24.parameters())
        print(f"  Parameters: {n24:,}")
        results["E24"] = runner.run_experiment(m24, "E24_D1024_8k", n_steps=8_000)
        _save_v4_results(results)

        print("\n  [E24 FINAL GENERATION SHOWCASE]")
        gen_r24 = gen_showcase(m24, "E24")
        results["E24"]["generation_ext"] = gen_r24
        del m24; gc.collect(); torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
        _save_v4_results(results)

    # ── E25: D=512 Continuous-context training ────────────────────────────────
    # Added after E21 proved continuous training 2.86× faster than reset-based.
    # This directly compares continuous vs reset-based at D=512, same scale as
    # the Round 3 champion (E20).
    if _want_exp("E25", results, only_train):
        print("\n\n>>> E25: D=512 CONTINUOUS-CONTEXT TRAINING (4k steps, compare vs E20)")
        params = ModelParams(state_size=512, num_partitions=8)
        cfg    = V3Config(label="E25_continuous_d512",
                          pde_type="spectral", hebbian_type="sparse",
                          partition_type="multiscale", resonance_type="none",
                          N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                          k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
        m25 = make_model(params, cfg)
        n25 = sum(p.numel() for p in m25.parameters())
        print(f"  Parameters: {n25:,}")

        # shuffle_docs=True interleaves text/code/math to prevent domain-transition bumps
        trainer25 = ContinuousCorpusTrainer(train_texts, m25, shuffle_docs=True)
        train_r25 = trainer25.train(n_steps=4_000, save_path=OUT_DIR/"E25_ContinuousD512.pt")

        print("\n  [Perplexity eval - cold state]")
        ppl_cold25 = eval_ppl_continuous(m25, eval_set, warm_tokens=0)
        print(f"    Macro NLL (cold) = {ppl_cold25.get('macro_avg_nll', 0):.3f}")

        print("\n  [Perplexity eval - stream-matched warm]")
        ppl_r25 = eval_ppl_continuous(m25, eval_set, warm_tokens=128)
        for mod, stats in ppl_r25.items():
            if isinstance(stats, dict):
                print(f"    {mod:<8}  nll={stats['avg_nll']:.3f}  ppl={stats['ppl']:,.0f}")
        print(f"    Macro NLL = {ppl_r25.get('macro_avg_nll', 0):.3f}")
        print(f"    Macro PPL = {ppl_r25.get('macro_ppl', 0):.1f}")

        print("\n  [Generation (with warm-up prefix)]")
        gen_r25 = gen_showcase_warm(m25, train_texts[:10])

        results["E25"] = {
            "info":       {"label": "E25_continuous_d512_shuffled", "n_params": n25},
            "train":      train_r25,
            "ppl_cold":   ppl_cold25,
            "ppl":        ppl_r25,
            "generation": gen_r25,
        }
        del m25; gc.collect(); torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
        _save_v4_results(results)

    # ── E26: Resume E25 → 8k total continuous steps (explicit --train E26) ────
    if _want_e26(only_train):
        ck25 = OUT_DIR / "E25_ContinuousD512.pt"
        if not ck25.exists():
            print("\n  [SKIP E26] Need checkpoint from E25 first:", ck25)
        else:
            print("\n\n>>> E26: RESUME E25 + 4k CONTINUOUS STEPS (8k total, D=512, shuffle_docs)")
            params = ModelParams(state_size=512, num_partitions=8)
            cfg    = V3Config(label="E26_continuous_d512_8k",
                              pde_type="spectral", hebbian_type="sparse",
                              partition_type="multiscale", resonance_type="none",
                              N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                              k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
            m26 = make_model(params, cfg)
            n26 = sum(p.numel() for p in m26.parameters())
            ck  = torch.load(str(ck25), map_location=DEVICE)
            m26.load_state_dict(ck["state_dict"])
            print(f"  Parameters: {n26:,}  (loaded from E25)")

            trainer26 = ContinuousCorpusTrainer(train_texts, m26, shuffle_docs=True)
            train_r26 = trainer26.train(n_steps=4_000, save_path=OUT_DIR/"E26_ContinuousD512_8k.pt")

            print("\n  [Perplexity eval - cold state]")
            ppl_cold26 = eval_ppl_continuous(m26, eval_set, warm_tokens=0)
            print(f"    Macro NLL (cold) = {ppl_cold26.get('macro_avg_nll', 0):.3f}")

            print("\n  [Perplexity eval - stream-matched warm]")
            ppl_r26 = eval_ppl_continuous(m26, eval_set, warm_tokens=128)
            for mod, stats in ppl_r26.items():
                if isinstance(stats, dict):
                    print(f"    {mod:<8}  nll={stats['avg_nll']:.3f}  ppl={stats['ppl']:,.0f}")
            print(f"    Macro NLL = {ppl_r26.get('macro_avg_nll', 0):.3f}")
            print(f"    Macro PPL = {ppl_r26.get('macro_ppl', 0):.1f}")

            print("\n  [Generation (with warm-up prefix)]")
            gen_r26 = gen_showcase_warm(m26, train_texts[:10])

            results["E26"] = {
                "info":       {"label": "E26_continuous_d512_8k", "n_params": n26,
                               "note": "4k steps resumed from E25_ContinuousD512.pt; 8k continuous total"},
                "train":      train_r26,
                "ppl_cold":   ppl_cold26,
                "ppl":        ppl_r26,
                "generation": gen_r26,
            }
            del m26; gc.collect(); torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
            _save_v4_results(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ROUND 4 SUMMARY (E21-E26)")
    print("=" * 72)

    # Full progression table from all rounds
    all_exp_results = {}
    for fn in ["results_v3.json", "results_v2.json"]:
        p = OUT_DIR / fn
        if p.exists():
            with open(p) as f:
                all_exp_results.update(json.load(f))
    all_exp_results.update(results)

    print(f"\n{'Exp':<6} {'Config':<38} {'D':>5} {'Steps':>6} {'MacroPPL':>12}")
    print("-" * 72)
    summary_rows = [
        ("E0",  "Dense baseline",                   256, 2000,  238_000_000),
        ("E8",  "Spectral+Multi",                   256, 2000,  11_083),
        ("E14", "Spectral+Multi D=512",              512, 4000,  498),
        ("E16", "Spectral+Multi+Sparse D=256",       256, 2000,  863),
        ("E18", "Spectral+Multi D=512 (8k)",         512, 8000,  322),
        ("E20", "Spectral+Multi+Sparse D=512 (8k)",  512, 8000,  246.6),
    ]
    for exp, cfg_label, d, steps, ref_ppl in summary_rows:
        r = all_exp_results.get(exp, {})
        ppl = r.get("ppl", {}).get("macro_ppl", ref_ppl) if r else ref_ppl
        if isinstance(ppl, str):
            try: ppl = float(ppl)
            except: ppl = ref_ppl
        print(f"{exp:<6} {cfg_label:<38} {d:>5} {steps:>6} {ppl:>12.1f}")

    print()
    for exp in ["E21", "E22", "E23", "E24", "E25", "E26"]:
        r = results.get(exp, {})
        if not r: continue
        tr  = r.get("train", {})
        ppl = r.get("ppl", {}).get("macro_ppl", "—")
        ini = tr.get("initial", 0)
        fin = tr.get("final", 0)
        lbl = r.get("info", {}).get("label", exp)
        if isinstance(ppl, float): ppl = f"{ppl:.1f}"
        print(f"{exp:<6} {str(lbl):<38} init={ini:.1f} final={fin:.3f} ppl={ppl}")

    results_path = OUT_DIR / "results_v4.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults -> {results_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
