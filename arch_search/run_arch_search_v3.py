"""
CellularAI v3 — Guided Architecture Search  Round 3  (E15–E21)
==============================================================
Addresses all weaknesses found in E8-E14 analysis:

Root-cause findings from Round 2:
  1. BUG: train_step_sequential bypassed extensions (resonance/lattice/kuramoto)
     by manually calling partitions/memory/metaplasticity instead of cellular_step().
     Fixed: train_step_sequential now calls self.cellular_step(inp).
  2. BUG: log_alpha_ext gate (exp(-2)=0.135) suppresses all extensions to zero
     during early training. Optimizer drives it to -inf. Fixed: removed entirely.
  3. BUG: PerFreqResonance gate_proj.bias=0 → sigmoid(0)=0.5 at init, drifts to 0.
     Fixed: gate_proj.bias initialised to +3.0 (sigmoid(3)=0.95).
  4. ISSUE: SparseHebbian hebb_rate=0.001 too small for fast warmup.
     Fixed: hebb_rate=0.005 (5×).
  5. STRUCTURAL: Router accuracy=0.333 (random) because cellular dynamics wash out
     domain markers. Fixed: domain tokens prepended to inputs.

Experiments:
  E15: SpectralPDE + MultiScale + PerFreqResonance (with ALL fixes)
       Baseline from E8 + resonance now actually trained
  E16: SpectralPDE + MultiScale + SparseHebbian (with hebb_rate fix)
       E10 revisited with 5× faster Hebbian warmup
  E17: SpectralPDE + MultiScale + Domain tokens (routing fix)
       Tests whether domain token prepending fixes routing accuracy
  E18: SpectralPDE + MultiScale D=512, 8000 steps (extended training)
       Fair comparison to E14 with 2× more steps
  E19: Best D=256 combined — E15 + domain tokens
       SpectralPDE + MultiScale + PerFreqResonance + domain tokens, D=256
  E20: E19 recipe at D=512, 8000 steps (champion)
       Maximum scale + all fixes

Results saved to data/local/arch_search/results_v3.json
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

MODALITY_NAMES = ["text", "code", "math"]
# Domain token prefixes — distinctive strings not in natural text/code/math
DOMAIN_PREFIXES = {
    0: "<<TEXT>> ",
    1: "<<CODE>> ",
    2: "<<MATH>> ",
}

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print("=" * 72)
print("CELL AI v3 — ARCH SEARCH ROUND 3 (E15-E20)")
print("=" * 72)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


from arch_search.run_arch_search import build_dataset, ExperimentRunner

from cellai_core.base    import ModelParams
from cellai_core.routing import AnnealedRouter
from v3.cell_ai_v3       import CellAIv3, V3Config


def make_model(params: ModelParams, cfg: V3Config) -> CellAIv3:
    return CellAIv3(params, cfg).to(DEVICE)


def build_domain_dataset(n_per: int = 6_000) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]:
    """Build dataset WITH domain token prefixes for E17/E19/E20."""
    raw = build_dataset(n_per)
    tagged = [(DOMAIN_PREFIXES[label] + text, label) for text, label in raw]
    n_train = int(len(tagged) * 0.85)
    return tagged[:n_train], tagged[n_train:]


# ─────────────────────────────────────────────────────────────────────────────
# E17 / E19: Domain-token routing experiment helper
# ─────────────────────────────────────────────────────────────────────────────

def run_with_routing(
    backbone: CellAIv3,
    train_set: List[Tuple[str,int]],
    eval_set:  List[Tuple[str,int]],
    n_steps:   int,
    label:     str,
) -> Dict:
    """Train backbone+router on domain-tagged data; eval routing accuracy."""
    from cellai_core.routing import AnnealedRouter

    n_mod  = len(MODALITY_NAMES)
    D      = backbone.params.state_size

    router = AnnealedRouter(D, n_modalities=n_mod,
                            T_start=2.0, T_end=0.5, anneal_steps=n_steps//3,
                            lambda_start=0.1, lambda_end=0.001).to(DEVICE)

    heads = nn.ModuleList([
        nn.Linear(D, D, bias=True) for _ in range(n_mod)
    ]).to(DEVICE)

    trainable = (list(backbone.parameters()) +
                 list(router.parameters()) +
                 list(heads.parameters()))
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1.5e-5)

    data_iter  = iter(train_set)
    ntp_losses, r_losses = [], []
    LOG_EVERY  = max(n_steps // 10, 1)

    print(f"  [Training backbone+router for {n_steps} steps]")
    for step in range(1, n_steps + 1):
        try:
            text, dom = next(data_iter)
        except StopIteration:
            data_iter = iter(train_set)
            text, dom = next(data_iter)

        opt.zero_grad()

        # Reset state (works for both single and multiscale partition managers)
        backbone.partitions.reset()
        backbone.memory_formation.reset()

        # NTP via routed heads (48 tokens)
        toks = backbone.encoder.tokenize(text)[:256]
        nll_val = 0.0
        if len(toks) >= 2:
            tok_ids  = torch.tensor(toks, dtype=torch.long, device=DEVICE)
            seg_len  = min(len(toks) - 1, 48)
            embs     = backbone.encoder.embedding(tok_ids[:seg_len+1]) * backbone.encoder._scale
            seg_loss = torch.tensor(0.0, device=DEVICE)
            count    = 0
            for t in range(seg_len):
                st   = backbone.cellular_step(embs[t])
                wts, _, _ = router(st.detach(), training=True)
                head_outs  = torch.stack([h(st) for h in heads], dim=0)
                routed     = (wts.unsqueeze(1) * head_outs).sum(dim=0)
                logits_ntp = routed @ backbone.encoder.embedding.weight.t()
                seg_loss   = seg_loss + F.cross_entropy(
                    logits_ntp.unsqueeze(0), tok_ids[t+1].unsqueeze(0))
                count += 1
            if count > 0:
                nll_val  = seg_loss.item() / count
                ntp_losses.append(nll_val)

        # Router classification (short prefix, fresh graph)
        backbone.partitions.reset()
        enc        = backbone.encode_input(text[:200])
        cell_state = backbone.cellular_step(enc)
        _, logits_r, bal_loss = router(cell_state.detach(), training=True)
        target     = torch.tensor([dom], dtype=torch.long, device=DEVICE)
        l_router   = F.cross_entropy(logits_r.unsqueeze(0), target)

        ntp_part = (seg_loss / max(count, 1)) if count > 0 else torch.tensor(0.0, device=DEVICE)
        total    = ntp_part + 0.5 * l_router + router.lambda_bal * bal_loss
        total.backward()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sch.step(); router.step_temperature()
        r_losses.append(l_router.item())

        if step % LOG_EVERY == 0:
            n_avg = sum(ntp_losses[-50:]) / max(len(ntp_losses[-50:]), 1)
            r_avg = sum(r_losses[-50:]) / max(len(r_losses[-50:]), 1)
            print(f"    step={step:>5}  ntp={n_avg:.3f}  router={r_avg:.3f}  "
                  f"H={router.routing_entropy():.3f}  T={router.temperature:.2f}  "
                  f"lam={router.lambda_bal:.4f}")

    # Routing accuracy
    print("  [Routing accuracy evaluation]")
    correct, total_ev = 0, 0
    with torch.no_grad():
        for text, label in eval_set[:400]:
            enc   = backbone.encode_input(text[:200])
            state = backbone.cellular_step(enc)
            _, logits_r, _ = router(state, training=False)
            if logits_r.argmax().item() == label:
                correct += 1
            total_ev += 1
    acc = correct / max(total_ev, 1)
    ent = router.routing_entropy()
    print(f"    Routing accuracy: {acc:.3f}  (random={1/n_mod:.3f})")
    print(f"    Router entropy:   {ent:.3f}  (max={math.log(n_mod):.3f})")

    return {
        "label":      label,
        "routing":    {"accuracy": acc, "entropy": ent},
        "ntp_final":  sum(ntp_losses[-100:]) / max(len(ntp_losses[-100:]), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────="────────
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    print("\n--- Loading data ---")
    plain_dataset = build_dataset(n_per=6_000)
    runner        = ExperimentRunner(plain_dataset)

    print("--- Loading domain-tagged data ---")
    dom_train, dom_eval = build_domain_dataset(n_per=6_000)
    print(f"  Domain train: {len(dom_train)}  eval: {len(dom_eval)}")

    # ── E15: SpectralPDE + MultiScale + PerFreqResonance (ALL BUGS FIXED) ─────
    print("\n\n>>> E15: SPECTRAL + MULTISCALE + PER-FREQ RESONANCE (all bugs fixed)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E15_spectral_multi_resonance_fixed",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="per_freq",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8,
                      gen_rep_penalty=1.3, gen_noise_std=0.03)
    m15 = make_model(params, cfg)
    n15 = sum(p.numel() for p in m15.parameters())
    print(f"  Parameters: {n15:,}")
    results["E15"] = runner.run_experiment(m15, "E15_SpectralMultiResonanceFixed")
    del m15; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E16: SpectralPDE + MultiScale + SparseHebbian (hebb_rate fix) ─────────
    print("\n\n>>> E16: SPECTRAL + MULTISCALE + SPARSE HEBBIAN (5x hebb_rate fix)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E16_spectral_multi_sparse_fixed",
                      pde_type="spectral", hebbian_type="sparse",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8,
                      k_frac=0.125, gen_rep_penalty=1.3, gen_noise_std=0.03)
    m16 = make_model(params, cfg)
    n16 = sum(p.numel() for p in m16.parameters())
    print(f"  Parameters: {n16:,}")
    results["E16"] = runner.run_experiment(m16, "E16_SpectralMultiSparseFixed")
    del m16; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E17: SpectralPDE + MultiScale + Domain tokens (routing fix) ───────────
    print("\n\n>>> E17: DOMAIN-TOKEN ROUTING (<<TEXT>>/<<CODE>>/<<MATH>> prepended)")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E17_domain_token_routing",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8)
    m17 = make_model(params, cfg)

    # First train backbone on domain-tagged data for NTP
    dom_runner = ExperimentRunner(list(zip(
        [t for t, _ in dom_train + dom_eval],
        [l for _, l in dom_train + dom_eval]
    )))
    dom_runner.train(m17)
    print("  [Backbone pre-training done]")

    # Then train router
    r17 = run_with_routing(m17, dom_train, dom_eval, n_steps=2000, label="E17_routing")
    results["E17"] = r17
    del m17; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E18: SpectralPDE + MultiScale D=512, 8000 steps ───────────────────────
    print("\n\n>>> E18: SPECTRAL + MULTISCALE D=512, 8000 STEPS (extended training)")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E18_spectral_multi_d512_8k",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="none",
                      N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                      gen_rep_penalty=1.3, gen_noise_std=0.03)
    m18 = make_model(params, cfg)
    n18 = sum(p.numel() for p in m18.parameters())
    print(f"  Parameters: {n18:,}")
    results["E18"] = runner.run_experiment(m18, "E18_SpectralMultiD512_8k", n_steps=8000)

    # Save E18 generation results
    print("  [E18 Generation test]")
    for prompt in ["Solve for x: 2x + 5 = 13",
                   "def fibonacci(n):",
                   "The transformer architecture learns"]:
        gen = m18.generate(prompt, max_tokens=40, temperature=0.85)
        toks = m18.encoder.tokenize(gen[len(prompt):])
        div  = len(set(toks)) / max(len(toks), 1)
        print(f"    '{prompt[:40]}' -> '{gen[len(prompt):len(prompt)+60]}'  [div={div:.2f}]")

    del m18; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E19: Best D=256 — E15 + domain tokens ─────────────────────────────────
    print("\n\n>>> E19: BEST D=256 — SPECTRAL + MULTISCALE + RESONANCE + DOMAIN TOKENS")
    params = ModelParams(state_size=256, num_partitions=4)
    cfg    = V3Config(label="E19_best_d256_domaintok",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="per_freq",
                      N_fast=4, D_fast=128, N_slow=2, D_slow=256, K_slow=8,
                      gen_rep_penalty=1.3, gen_noise_std=0.03)
    m19 = make_model(params, cfg)
    dom_runner19 = ExperimentRunner(dom_train + dom_eval)
    results["E19_ntp"] = dom_runner19.run_experiment(m19, "E19_BestD256DomainTok")

    # Also test routing on E19
    r19 = run_with_routing(m19, dom_train, dom_eval, n_steps=2000, label="E19_routing")
    results["E19_routing"] = r19
    del m19; gc.collect(); torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── E20: E19 recipe at D=512, 8000 steps (champion) ──────────────────────
    print("\n\n>>> E20: CHAMPION — BEST RECIPE D=512, 8000 STEPS + DOMAIN TOKENS")
    params = ModelParams(state_size=512, num_partitions=8)
    cfg    = V3Config(label="E20_champion_d512_8k",
                      pde_type="spectral", hebbian_type="full",
                      partition_type="multiscale", resonance_type="per_freq",
                      N_fast=4, D_fast=256, N_slow=2, D_slow=512, K_slow=8,
                      gen_rep_penalty=1.3, gen_noise_std=0.03)
    m20 = make_model(params, cfg)
    n20 = sum(p.numel() for p in m20.parameters())
    print(f"  Parameters: {n20:,}")

    dom_runner20 = ExperimentRunner(dom_train + dom_eval)
    results["E20_ntp"] = dom_runner20.run_experiment(m20, "E20_ChampionD512", n_steps=8000)

    r20 = run_with_routing(m20, dom_train, dom_eval, n_steps=2000, label="E20_routing")
    results["E20_routing"] = r20

    # Final generation showcase
    print("\n  [E20 FINAL GENERATION SHOWCASE]")
    prompts = [
        ("math", "Solve for x: 3x^2 - 12 = 0"),
        ("math", "Integrate x^2 from 0 to 1"),
        ("code", "def binary_search(arr, target):"),
        ("code", "class LinkedList:\n    def __init__(self):"),
        ("text", "The cellular automaton model predicts that"),
        ("text", "Neural networks learn representations by"),
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
    print("ROUND 3 ARCHITECTURE SEARCH SUMMARY (E15-E20)")
    print("=" * 72)
    print(f"{'Exp':<6} {'Label':<38} {'Init':>8} {'Final':>8} {'PPL(mac)':>14} {'ms/call':>8}")
    print("-" * 72)
    for exp in ["E15","E16","E18","E19_ntp","E20_ntp"]:
        r = results.get(exp, {})
        if not r:
            continue
        tr  = r.get("train", {})
        ppl = r.get("ppl",{}).get("macro_ppl", "-")
        ms  = r.get("throughput", {}).get("full_ms", "-")
        ini = tr.get("initial", 0)
        fin = tr.get("final", 0)
        lbl = r.get("info",{}).get("label", exp)
        if isinstance(ppl, float): ppl = f"{ppl:.1f}"
        if isinstance(ms,  float): ms  = f"{ms:.3f}"
        print(f"{exp:<6} {str(lbl):<38} {ini:>8.1f} {fin:>8.3f} {str(ppl):>14} {str(ms):>8}")

    for exp_r in ["E17","E19_routing","E20_routing"]:
        r = results.get(exp_r, {})
        if r:
            acc = r.get("routing", {}).get("accuracy", "-")
            ent = r.get("routing", {}).get("entropy", "-")
            ntp = r.get("ntp_final", "-")
            if isinstance(acc, float): acc = f"{acc:.3f}"
            if isinstance(ent, float): ent = f"{ent:.3f}"
            if isinstance(ntp, float): ntp = f"{ntp:.3f}"
            print(f"{exp_r:<6} acc={acc}  entropy={ent}  ntp={ntp}")

    # Save results
    results_path = OUT_DIR / "results_v3.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results -> {results_path}")
    print("DONE.")
    return results


if __name__ == "__main__":
    main()
