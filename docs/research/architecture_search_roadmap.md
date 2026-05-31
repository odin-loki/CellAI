# Architecture search roadmap (post E0–E26)

This document is the **working plan** for future guided architecture search on CellularAI v3. It complements **`docs/ARCH_SEARCH_PAPER.md`** (results through §19) and is meant to be updated as waves complete.

**Goals:** stronger **science** (what we measure and conclude), **engineering** (how we run and compare runs), and **architecture** (what we try next).

---

## 1. North star

### 1.1 Primary metrics (by training regime)

| Regime | Headline metric | Secondary |
|--------|-----------------|-----------|
| **Reset-based** (e.g. E20 / E22 / E23 / E24) | **Text+code** held-out NLL / PPL | Per-domain NLL, macro PPL (note: math is easy—macro is optimistic) |
| **Continuous** (e.g. E21 / E25 / E26) | **Stream-matched warm** NLL / PPL with fixed **BPTT detach cadence + stream burn** | Cold PPL only as OOD diagnostic, not headline |

Every new experiment row in results JSON should record **protocol version** (burn length, detach period, `n`, seed).

### 1.2 Champion discipline

Maintain two live champions when both regimes are active:

- **Reset champion** — default comparison for scaling and head/router changes.
- **Continuous champion** — default comparison for long-context and stream training.

Promotion requires passing the **same eval protocol** as the incumbent (or a strictly newer, documented protocol).

### 1.3 Pre-registration (lightweight)

Before a wave starts, fix in writing:

- Step / compute budget and hardware class.
- LR schedule family and search bounds (if any).
- Checkpoint rule: **best vs last** on which score.
- Early-stopping rule (if any) and which **held-out** signal it uses.

This reduces ambiguous reads like “more continuous steps regressed” without knowing optimizer state, LR, or selection metric.

---

## 2. Phase A — Metrics and evaluation (foundation)

**Do this before large sweeps.**

| Initiative | Rationale |
|------------|-----------|
| **Eval config object** | Single source of truth (YAML/JSON): `burn_tokens`, `detach_every`, `warm_tokens`, eval `n`, seed, tokenizer id; embed hash or version id in `results_v*.json`. |
| **Text+code leaderboard** | Promote `text_code_avg_nll` / PPL alongside macro; macro remains for continuity with E0–E26 tables. |
| **Held-out stream for continuous** | Fixed held-out stream slice (or second shuffle seed) used **only** for model selection / early stopping—not identical to training stream statistics. |
| **Generation spot checks** | Small fixed prompt suite + diversity; optional small human or LLM-judge panel for **top** checkpoints only. |
| **OOD spot checks** | Occasional eval on alternate held-out slice to detect overfitting to one mixture. |

---

## 3. Phase B — Training dynamics (high ROI)

Addresses open gaps from Round 4 (e.g. E24 vs E22; E26 vs E25) without new modules first.

| Track | Directions (future experiment ids, e.g. E27+) |
|--------|-----------------------------------------------|
| **D=1024, long reset** | 16k–32k steps, cosine LR, match E22 **philosophy**; success = beat **E22 on text+code**, not macro alone. |
| **Continuous continuation** | EMA weights; **lower LR** on resume; ablate **fresh optimizer vs continue**; gradient clipping sweep. |
| **Early stopping** | Continuous: stop on held-out **warm** text+code; reset: stop on held-out text+code; save **best** and **last**. |
| **Regularization** | Weight decay; small dropout on heads; **Hebbian rate** schedule (decay after warmup). |

---

## 4. Phase C — Architecture hypotheses (v3.2+)

Aligned with **`docs/ARCH_SEARCH_PAPER.md`** §15 “Next generation (v3.2) targets” and §19.

| Theme | Hypothesis | Sketch |
|--------|------------|--------|
| **Domain-specific dynamics** | Token-level / cellular-state routing stays near-random because diffusion washes modality; **branching computation** may work. | **PDE banks** or **low-rank adapters** on SpectralPDE / slow partitions, selected by lightweight **input-side** routing (not sole reliance on cellular-state softmax). |
| **Bypass pathways** | Modality signal must survive PDE mixing. | Shallow **encoder→head skip** or parallel trunk merged with aggregated cellular state. |
| **Document-level signal** | NTP alone underpins coherence limits. | Small **contrastive** or **next-chunk** loss on pooled state alongside NTP. |
| **PerFreqResonance at scale** | Failed at D=256; might matter when more frequency bins matter. | Controlled on/off at **D≥1024**, **text+code** headline only. |
| **Decoding** | Decode settings confound architecture comparisons. | Fixed prompt suite + sweep rep penalty, noise, temperature, nucleus for **final** showcases only. |

---

## 5. Phase D — Process and tooling

| Piece | Purpose |
|--------|---------|
| **Experiment registry** | `experiments.yaml` (or similar): id, hypothesis, command, data deps, expected artifacts—feeds CLI help and paper tables. |
| **Run harness** | One wrapper: seeds, `git` hash, protocol version, writes `meta.json` beside checkpoints. |
| **Resume / fork** | Standard “load checkpoint X, apply delta config” (generalize E26-style resume). |
| **Hyperband / successive halving** | Screen many short runs on LR / steps / hebb_rate; promote top configs to full budget. |
| **NAS-lite** | Random search + top-k validation over **bounded** knob sets; human team defines **which** knobs enter the space. |

Implementation home: **`arch_search/`** modules and `data/local/arch_search/` artifacts; keep parity with **`python -m arch_search.*`** from repo root.

---

## 6. Phase E — Reporting

- **Paper:** add a “Round 5+ protocol” subsection when the first post-E26 wave uses a new eval version; keep E0–E26 frozen as historical.
- **Failure log:** short bullets (hypothesis, outcome, plausible cause)—as important as wins.
- **Cost log:** approximate GPU-hours per experiment class for budgeting.

---

## 7. Suggested waves (ordering)

| Wave | Focus | Outcome |
|------|--------|---------|
| **1** | Phase A + JSON/schema for protocol version; best+last checkpoints | Trustworthy comparison of all future runs. |
| **2** | Phase B: D=1024 long reset vs E22; continuous forks of E25 with EMA/LR/stop | Clear scaling and continuation story. |
| **3** | Phase C: one domain-bank or bypass design vs champion | Test routing/coherence hypotheses. |
| **4** | Phase D hyperband on champion skeleton only | Efficiency; avoid full-grid explosion. |

---

## 8. Success criteria (examples)

- **Reset:** Match or beat **E22** on **text+code** at declared eval `n` and protocol, or document a negative result with controlled ablations (steps, LR, width).
- **Continuous:** Match or beat **E25** on **stream-matched warm text+code** with pre-registered burn/detach.
- **Routing:** Held-out routing accuracy **above chance + margin**, or narrow the claim (“routing from cellular state” vs “routing from inputs”).
- **Generation:** Prompt suite shows intended trade-off (e.g. less repetition) without hiding PPL regression.

---

## 9. References in-repo

- Results and narrative: **`docs/ARCH_SEARCH_PAPER.md`** (§15 recommendation, §17–§18 Round 4, §19 limitations).
- Cross-architecture context: **`docs/CELLULARAI_PAPER.md`**.
- Runners: **`arch_search/`**, **`scripts/README.md`**.

---

*Last aligned with ARCH_SEARCH narrative through Round 4 (E21–E26). Revise this file when Wave 1 protocol versioning lands.*
