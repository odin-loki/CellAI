# Cell AI — biology-motivated sequence-modelling architecture (CellularAI)

> **An experimental, biology-motivated sequence-modelling architecture that replaces self-attention with reaction-diffusion-style partition dynamics, online Hebbian / BCM plasticity that runs *during the forward pass* (not after), optional spectral and multi-scale extensions, and multi-domain routing — culminating in a v3 architecture-search programme (E0 through E26) whose best run reports macro-perplexity `246.6` for an asserted `966 000 ×` improvement over the E0 baseline (the framing is the author's, the metric is internal).** The repository is honest about what it is: **`125.8 M`-parameter v1** does not approach transformer perplexity on real corpora (paper reports training PPL bands of `450 k – 1.18 M` against GPT-2's `~20`), the in-forward Hebbian update introduces gradient-starvation and BPTT issues that the v2 / v3 work documents and partially fixes, and the v3 architecture-search ladder includes a measured *regression* (E26 macro PPL `~2 002` worse than E25's `~1 379`). The interesting move is not "we beat transformers" — it is "what does a *fully different* sequence model look like, what fails, what works, and can we close the gap honestly?"

---

## What this folder is

The dominant sequence-model architecture of 2025 is the transformer, and the question this folder takes seriously is: *what does a fundamentally different architecture look like at full scale?* CellularAI's answer is biology-motivated. The core operator is a **CellularPDE** that maintains `N = 4` partitions of `D = 256`-dimensional state, with leakage `λ = 0.01` and a sigmoid-based reaction term governed by learned matrices — analogous to a reaction-diffusion system on a discrete lattice. On top of this sit a **MetaplasticityLayer** doing BCM-style sliding-threshold Hebbian learning (`α = 0.1, β = 0.01`), a **MemoryFormation** module, **ResonanceSystem** (FFT-phase coupling), **CrystalLattice** (`K = 3` order, 27-site Tucker-rank-1 sums), and Kuramoto-style oscillator coupling in v2. The router from the **MultiModalModel** uses a load-balanced loss `L = λ_r · L_router + λ_ntp · L_ntp` with `λ_r = 1, λ_ntp = 0.3`, vocabulary `cl100k_base` (`V = 100 277`).

The v2 / v3 / arch-search line of work is the more substantive part of the folder. v3 introduces a **SpectralPDE** that drops complexity from `O(D²)` to `O(D log D)`, a **SparseHebbian** layer that goes from `D² = 65 536` updates to `k² = 1024` at top-`k = D/8`, **MultiScalePartitions** with fast (`N_f = 4, D_f = 128`) and slow (`N_s = 2, D_s = 256` every `K = 8` tokens) streams, and an **AnnealedRouter** with Gumbel-Softmax + load-balance penalty `λ_bal = 0.02`. The arch-search paper documents the entire E0–E26 experiment ladder with explicit success / failure annotation — including the named failure modes (PerFreqResonance, routing collapse near random, E26 underperforming E25). The v3 narrative also gestures toward a "Cell-Fungal Harmonic" integration that the paper labels speculative / planned.

This is closer to a research log than a product README. Read it for the architecture and the bug archaeology.

---

## 📑 Source documents

### Top-level

| File | Role |
|---|---|
| [`docs/CELLULARAI_PAPER.md`](docs/CELLULARAI_PAPER.md) | Main research paper. CellularPDE, MetaplasticityLayer, MultiModalModel, v1 + v2 architecture, throughput tables, gradient-norm diagnostics. |
| [`docs/ARCH_SEARCH_PAPER.md`](docs/ARCH_SEARCH_PAPER.md) | Architecture-search programme paper. **E0 – E26** experiment ladder. SpectralPDE, SparseHebbian, MultiScalePartitions, AnnealedRouter. Honest failure annotations (PerFreqResonance, routing collapse, E26 regression). |
| [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md) | Evaluation report. |
| [`docs/README.md`](docs/README.md) | Documentation index. |

### Architecture deep-dives

| File | Role |
|---|---|
| [`docs/architecture/v2_intro.md`](docs/architecture/v2_intro.md) | v2 architectural overview. |
| [`docs/architecture/v2_math_model.md`](docs/architecture/v2_math_model.md) | v2 mathematical model. |
| [`docs/architecture/v2_vs_v1.md`](docs/architecture/v2_vs_v1.md) | v2 vs v1 comparison. |
| [`docs/architecture/v3_architecture.md`](docs/architecture/v3_architecture.md) | v3 OICFHS architecture. |
| [`docs/architecture/v3_math_model.md`](docs/architecture/v3_math_model.md) | v3 mathematical model. |
| [`docs/architecture/multimodal_architecture.md`](docs/architecture/multimodal_architecture.md) | Multi-modal architecture document. |
| [`docs/architecture/parallel_model.md`](docs/architecture/parallel_model.md) | Parallel model. |
| [`docs/architecture/thinking_cot_readme.md`](docs/architecture/thinking_cot_readme.md) | Chain-of-thought / reasoning module. |

### Math models, research notes

| File | Role |
|---|---|
| [`docs/math_models/core_math_model.md`](docs/math_models/core_math_model.md), [`software_math_model.md`](docs/math_models/software_math_model.md), [`math_domain_model.md`](docs/math_models/math_domain_model.md), [`nlp_math_model.md`](docs/math_models/nlp_math_model.md), [`advanced_techniques.md`](docs/math_models/advanced_techniques.md) | Mathematical model documents. |
| [`docs/research/`](docs/research/) | Research notes — `cell_ai_explanation.md`, `nlp_techniques_explanation.md`, `architecture_search_roadmap.md`, `computational_research_plan.md`, `research_plan.md`, `research_possibilities.md`, `brain_comparison_outline.md`, `answered_questions.md`. |

### Implementation

| File | Role |
|---|---|
| [`v1/cell_ai.py`](v1/cell_ai.py) | v1 reference implementation. |

(See also `cellai_core/` for shared base modules — `base.py` (`CellularPDE`), `spectral_pde.py`, `sparse_hebbian.py` — and `v2/`, `v3/`, `arch_search/` for evolution branches if present in your tree.)

---

## 🧠 The core architecture

```
Input tokens ──▶ Token embedding (cl100k_base, V = 100 277)
                       │
                       ▼
            ┌──── CellularPDE (N=4 partitions, D=256) ────┐
            │     leakage λ = 0.01                         │
            │     sigmoid reaction with learned matrices   │
            └──────────────────────────────────────────────┘
                       │
                       ▼
            ┌──── MetaplasticityLayer (BCM) ──────┐
            │     α = 0.1, β = 0.01                │
            │     Online Hebbian during forward    │
            └──────────────────────────────────────┘
                       │
                       ▼
            MemoryFormation  ─── ResonanceSystem (FFT phase)
                       │            │
                       ▼            ▼
                CrystalLattice (K=3, 27 sites, Tucker rank-1)
                       │
                       ▼
                  Kuramoto coupling (v2)
                       │
                       ▼
            MultiModalModel router
            L = λ_r · L_router + λ_ntp · L_ntp
            λ_r = 1.0,  λ_ntp = 0.3
                       │
                       ▼
                Output token logits
```

### v3 SpectralPDE / Sparse / Multi-scale

| Module | v1 / v2 cost | v3 cost |
|---|---|---|
| **PDE step** | `O(D²)` dense | **`O(D log D)`** SpectralPDE |
| **Hebbian update** | `D² = 65 536` weights | **`k² = 1 024` at top-k = D/8** SparseHebbian |
| **Streams** | single | **fast (N_f=4, D_f=128) + slow (N_s=2, D_s=256, every K=8 tokens)** MultiScalePartitions |
| **Routing** | static | **AnnealedRouter** Gumbel-Softmax + λ_bal = 0.02 |

---

## 📊 Reported metrics

### v1/v2 paper

| Metric | Value |
|---|---|
| Total parameters (v1) | `125.8 M` |
| Vocabulary | `cl100k_base`, `V = 100 277` |
| Multimodal NTP loss | `142.1 → 12.8 nats/token over 5 000 steps` |
| Routing accuracy | `21 % → 34.3 %` (with class collapse) |
| Forward latency (RTX 3090) | **`4.564 ms` full forward, `219 calls/s`** |
| Peak CUDA memory | `767.1 MB` |
| Post-state-gate-fix gradient norms | `10⁻⁴ – 10⁻³` (PDE) |

### Arch-search paper (E0 – E26)

| Experiment | Macro PPL | Notes |
|---|---|---|
| E0 baseline | (huge) | reference |
| **E20** | **`246.6`** | **`966 000 ×` vs E0** in author framing |
| E21 / E25 (warm) | `~1 379` | |
| **E26** | **`~2 002`** | **regression vs E25** (documented in paper) |

Continuous-training protocol: `partition detach every 64 tokens`, burn-in `~4 096`.

---

## 🚧 Honest caveats (paper §, explicit)

- **The main paper does NOT claim competitive perplexity vs transformers.** v1/v2 training PPL bands are `450 k – 1.18 M` vs GPT-2's `~20` baseline. This is a research artefact, not a competitive language model.
- **Router collapse** is documented — the multimodal router degenerates to favouring a small subset of classes.
- **Generation is NOT autoregressive in v1** — chat-style outputs have known pathologies.
- **v2 resonance phase shows minimal gradient.** The FFT-based ResonanceSystem barely contributes to learning.
- **Macro NLL is "optimistic (easy math)"** — the headline numbers in arch-search use a lenient evaluation that the paper labels as such.
- **Failure modes named in arch-search:** PerFreqResonance (didn't help), routing near random, E26 underperforming E25.
- **v3 "Cell-Fungal Harmonic"** integration in the v3 architecture document is speculative / planned, not implemented.

---

## 🎯 What this is genuinely interesting for

| Audience | Use |
|---|---|
| Sequence-model researcher | Worked example of "what does a non-attention architecture actually look like end-to-end?" |
| Compiler / kernel writer | SpectralPDE `O(D log D)` is a transferable optimisation |
| Plasticity researcher | In-forward Hebbian + BPTT-aware fixes |
| Anyone disillusioned with attention | An honest baseline for what fails when you swap it out |
| Anyone wanting GPT-class performance | Wrong folder — see established LLMs |

---

## 🔗 Related work in this repo

- [`../Cypha/`](../Cypha/) — HRNA inference + training stack (sister architecture)
- [`../Long Reasoning and Thinking NN/`](../Long%20Reasoning%20and%20Thinking%20NN/) — UHPM (long-context memory; complementary)
- [`../Compression Algorithms/`](../Compression%20Algorithms/) — NMP `α = 0.851 ± 0.122` spectral exponent (relates to architecture-search efficiency claims)
- [`../NN Shortcuts/`](../NN%20Shortcuts/) — Streaming Geometry Framework + Algebraic Autopsy (post-hoc decomposition that could analyse Cell AI weights)
- [`../GF2 Algebra and Applications/`](../GF2%20Algebra%20and%20Applications/) — DLGN gate-frequency learning (sister work)
- [`../Fungal Network Algorithm/`](../Fungal%20Network%20Algorithm/) — bio-inspired sister; "Cell-Fungal Harmonic" is the v3 integration story
- [`../Veritas/`](../Veritas/) — formal-verification framework (could in principle analyse Cell AI's Boolean function class)

---

[← Back to main README](../README.md)
