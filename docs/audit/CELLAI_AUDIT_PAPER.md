# CellAI Under the Microscope: An Independent Profile, Data-Mining Study and Audit of a Biology-Inspired Sequence Model

**Scope:** repository `odin-loki/CellAI` at commit `a109790` (4 commits, May–Sep 2026)
**Date:** 2026-11-03
**Artefacts:** everything referenced here lives in [`docs/audit/`](.): figures, raw data (`data/`) and the scripts that produced every number (`scripts/`).

---

## Abstract

CellularAI ("CellAI") is a research codebase proposing a non-attention language model built from biologically motivated parts:
- a reaction–diffusion "cellular PDE" over parallel state partitions;
- online Hebbian/BCM plasticity applied during the forward pass;
- FFT "resonance", crystal-lattice and Kuramoto-oscillator extensions;
- a v3 line found by a 36-experiment guided architecture search (E0–E35).

We profiled the code empirically on CPU, statically mined its 12k lines of Python and 12.6k lines of documentation, extracted every experiment into a single table, re-derived the evaluation splits, and built count-based baselines under the project's exact data and evaluation protocol.

**Main result.** On the held-out text+code metric the project itself adopted as its headline, the reported CellAI champions score 7.85–8.04 nats/token. A smoothed **unigram** model, which uses no context at all, scores **7.87–7.91** on the same documents. Kneser–Ney bigram and trigram models score **5.46–5.52** and **4.89–4.99**. The best model, E34 (7.30), gets its gain from a zero-initialised "bypass" that adds the current token's embedding directly to the output state, i.e. a learned bigram path around the cellular core.

A controlled training probe at matched parameter count agrees: no Cell variant beats a stateless bigram MLP.

Mechanistically:
- the v1/v2 dense PDE cannot respond to its input from the zero initial state at all;
- the v3 spectral PDE's input-dependent signal is about 0.2% of its state at initialisation;
- three of the v2 "extensions" reduce to a fixed scale factor, a fixed rank-27 linear map, and an input-independent bias.

We also find:
- **Evaluation contamination.** E22 and every Wave 2–4 experiment (E27–E35, including champion E34) are scored on an eval set about 85% of whose documents their ancestor checkpoints trained on. The cause is import-time RNG seeding that differs between scripts.
- **Headline figure errors.** The "966,000×" improvement compares a training loss to a held-out score; the like-for-like ratio is about 2,700×. The "125.8 M parameter" figure is really 26.0 M.
- **Unimplemented complexity claims.** The spectral PDE only beats a dense PDE at D ≥ 1024. The sparse Hebbian layer still forms a dense D×D outer product, so its claimed k² saving does not exist. About 94% of training time is spent in a per-token vocabulary projection that could be batched for a 5–8× speed-up.

The project documents its own failures unusually candidly, and several of its diagnoses are confirmed here. Its central empirical claim, that the cellular architecture learns useful context, is **not supported** by the evidence in the repository.

---

## 1 Introduction

CellAI asks a legitimate question: what does a sequence model built from biological mechanisms (diffusion, Hebbian plasticity, oscillators) look like end to end, and can it be made competitive? The repository contains three model generations (v1, v2, v3), a shared core library (`cellai_core/`), domain pipelines (math, NLP, code), and seven generations of architecture-search drivers (`arch_search/run_arch_search*.py`). There are three main write-ups:

- [`docs/CELLULARAI_PAPER.md`](../CELLULARAI_PAPER.md): v1/v2 architecture, training, and bug archaeology.
- [`docs/ARCH_SEARCH_PAPER.md`](../ARCH_SEARCH_PAPER.md): the E0–E35 search, SpectralPDE, SparseHebbian, MultiScale partitions, routing.
- [`docs/EVALUATION_REPORT.md`](../EVALUATION_REPORT.md) and [`docs/ARCH_SEARCH_FAILURE_LOG.md`](../ARCH_SEARCH_FAILURE_LOG.md).

These documents are unusually candid. They report negative results, name failure modes, and repeatedly warn that the macro metric is "optimistic (easy math)". This study takes that spirit further. We ask what the repository's own evidence supports when every number is checked against the code, and when the missing baselines are supplied.

**Contributions**

1. **Profile** (§4): parameter census, latency, scaling exponents and op-level profiles of v1, v2, v3 and the E20/E34 champion configurations.
2. **Data mining** (§5): a 38-row experiment table (E0–E35 plus diagnostics) with outcomes, a best-so-far trajectory, and a claims-vs-code audit.
3. **Baselines** (§6): unigram/bigram/trigram models under the exact data records, split and `arch_eval_v1` protocol, plus a parameter-matched training probe.
4. **Mechanistic analysis** (§7): why the cellular state carries little input information.
5. **Evaluation integrity** (§8): split contamination, test-set model selection, metric definitions.
6. **Engineering findings** (§9) and **recommendations** (§11).

---

## 2 System overview

### 2.1 Architecture as implemented

Every version shares the same skeleton. For each token $x_t$ (cl100k_base, $V = 100{,}277$):

1. Embed: $e_t = E[x_t]/\sqrt{D}$, with the embedding $E$ tied to the output.
2. Advance the partition state $S_t$ with a PDE step.
3. Aggregate the partitions to $a_t$.
4. Memory: $m_t = \tanh(W_m[e_t \,\|\, a_t])$, with inputs detached.
5. Hebbian layer: $h_t = (W_H e_t) \odot \sigma(G a_t + b)$, where $W_H$ is modified in place by an outer-product rule.
6. Output state: $o_t = W_o h_t$ (+ v2/v3 extensions, + optional bypass $\sigma(\alpha) P e_t$).
7. Logits: $o_t E^\top$, trained with cross-entropy using truncated BPTT over 64-token segments.

The architecture-search papers place the cellular state as a gate on a linear function of the current token. That reading is correct, and it matters for §7.

| Component | Where | What it actually computes |
|---|---|---|
| Dense `CellularPDE` (v1/v2/E0) | `cellai_core/base.py:79` | $S \leftarrow S + dt(\sigma(IW^\top)\odot\tanh(SE^\top) - \gamma S + D\nabla^2_{\text{ring}}S + \eta\varepsilon)$, $dt = 0.01$ |
| `SpectralPDE` (v3) | `cellai_core/spectral_pde.py` | $S \leftarrow S + dt(\sigma(\mathrm{irfft}(\hat S H_s + \hat I H_i)) - \gamma S + D_{\text{diff}}\nabla^2 S)$ |
| "Ring" Laplacian | both | `torch.roll(..., dims=0)`: diffusion across the 2–8 *partitions*, not across features |
| `MetaplasticityLayer` (full Hebbian) | `cellai_core/memory.py` | non-negative outer-product update to $W_H$ on every forward pass, clamped to ±1 |
| `SparseHebbian` | `cellai_core/sparse_hebbian.py` | top-k activity scattered into a dense vector, then a dense D×D outer product; deferred until after `backward()`; **off in `eval()`** |
| `MultiScalePartitions` | `cellai_core/multiscale.py` | fast stream every token, slow stream every K = 8 tokens, cross-links `slow_to_fast` (zero-init) and `fast_to_slow` |
| v2 Resonance | `v2/cell_ai_v2.py` | $\mathrm{Re}(\mathcal F^{-1}(\mathcal F(s)e^{i\phi})) = \cos\phi \cdot s$ exactly (verified to 4·10⁻⁷) |
| v2 CrystalLattice | `v2/cell_ai_v2.py` | fixed symmetric rank-27 linear map $\Phi^\top \mathrm{diag}(T)\Phi$, no non-linearity |
| v2 Kuramoto | `v2/cell_ai_v2.py` | phase detached each step, so `natural_freq` gets exactly zero gradient; an input-independent drifting bias |
| E34 bypass | `v3/cell_ai_v3.py:340` | $o_t \mathrel{+}= \sigma(\alpha)Pe_t$, zero-initialised, $\alpha_0 = -8$ |

### 2.2 The search programme

The 36 numbered experiments run in seven rounds/waves. The headline metric changed during the search:

- **Rounds 1–3 (E0–E20):** "macro NLL", the mean of text, code and math NLL.
- **Round 4 (E21–E26):** macro NLL plus a text+code caveat.
- **Waves 2–4 (E27–E35):** "text+code NLL" under protocol `arch_eval_v1` (`arch_search/config/eval_protocol_v1.json`): 50 documents per modality, first 256 tokens, reset state, per-document mean NLL averaged over documents.

The change happened because math NLL fell to about 0.5 nats on generator-templated problems and dominated the macro average.

---

## 3 Method

| Activity | How | Where |
|---|---|---|
| Runtime profiling | CPU (4 cores, 15 GB, no GPU), torch 2.15, median-of-N timings with warm-up, `torch.profiler` | `scripts/profiling/`, `data/profiling_results.json` |
| Static mining | LOC/complexity (radon), import graph, constructor defaults, `git log`, pytest + coverage | §5, §9 |
| Experiment table | every E-number pulled from the papers, configs and scripts, with the paper's own verdict normalised | `data/experiments.csv` |
| Split re-derivation | each `arch_search` driver imported in its natural import order, `build_dataset` called on uniquely labelled dummy records, permutations compared | `scripts/split_simulation.py` |
| Baselines | records rebuilt exactly as `scripts/run_full_pipeline.py` writes them (wikitext-103 train lines ≥ 50 chars; code_search_net Python with docstring prefix, 8k-char cap), first 6,000 per domain, the project's split permutation, `arch_eval_v1` scoring | `scripts/ngram_baselines.py`, `data/ngram_split_*.json` |
| Training probe | v1, v2, v3 variants vs a 2-layer Transformer, a bigram MLP and unigram, all at about 6.4 M parameters (D = 64), 128 k tokens of the repo's own docs | `scripts/profiling/train_probe.py` |

**Limitations of method.**
- No checkpoints, result JSONs or logs are committed (`data/local/` is git-ignored), so the reported numbers cannot be re-derived. We compare against them as published.
- Our record reconstruction uses the current HuggingFace parquet exports. Small differences from the author's local files are possible, but our training-token count (2.19 M text+code) matches the paper's corpus description (2.3 M including math).
- There is no GPU, so champion-scale training was not repeated. E20/E34 architectures were profiled at full size and trained at D = 64.

---

## 4 Profile

### 4.1 Parameters: almost everything is the embedding

| Model | D | Total params | Non-embedding | Embedding share |
|---|---|---|---|---|
| v1 default | 256 | 25,999,104 | 328,192 | 98.7% |
| v2 default | 256 | 26,007,075 | 336,163 | 98.7% |
| v3 default (`V3Config()` = v1 architecture) | 256 | 25,999,104 | 328,192 | 98.7% |
| E20 config | 512 | 53,775,650 | 2,433,826 | 95.5% |
| E34 config | 1024 | 109,054,265 | 6,370,617 | 94.2% |

![Parameter breakdown](figures/fig3_param_breakdown.png)

- The README and main paper state **125.8 M** ("25.5 M backbone + 100.3 M embedding"). The embedding is 100,277 × 256 = 25.67 M and the cellular backbone is 0.33 M. `EVALUATION_REPORT.md` has the correct figure (25,999,104). The E20/E23/E24 counts in the arch-search paper match ours.
- `SpectralPDE.input_gate` is a D×D `nn.Linear` that is created but never used in `step()`. At D = 512 it is 262,144 of the slow PDE's 265,228 parameters: about **98% of E20's "PDE" parameters are dead**. The paper's spectral parameter count (67,600 at D = 256) includes it. The parameters actually used are about 4F + 4NF (about 2.6 k at D = 256).

### 4.2 Latency and where time goes

| Model | Cell step (ms) | + vocab head (ms) | Inference tok/s | Train tok/s (repo method) |
|---|---|---|---|---|
| v1 default | 0.33 | 1.71 | 587 | 24.1 |
| v2 default | 0.72 | 2.26 | 442 | 23.6 |
| E20 config | 1.21 | 5.27 | 190 | 15.2 |
| E34 config | 1.20 | 10.65 | 94 | 7.8 |

All figures are CPU, 4 threads, median, ±20–30% container noise.

`torch.profiler` on one 32-token training segment shows:
- `aten::mm` takes 55–59% of time and gradient accumulation (`add_`) 16–25%;
- **FFT is under 1% of time**;
- the hot path is the tied vocabulary head, run once per token *inside* the Python time loop, each time doing a 100,277 × D weight-gradient matmul.

The head does not affect the recurrent state, and SparseHebbian updates are deferred until after `backward()`. So stacking the segment's states and running one (T×D)·(D×V) head gives the same loss and gradient, and is **4.8× faster (v1) to 7.7× faster (E20)**. This is the largest available optimisation in the codebase.

> The papers' "4.56 ms full forward / 219 calls/s" and "~130k tok/s" figures time **one** cellular step on a pooled encoding of a whole string, not per-token sequence processing.

### 4.3 Do the complexity claims materialise?

![PDE scaling](figures/fig4_pde_scaling.png)

| D | Dense fwd (ms) | Spectral fwd (ms) | Dense fwd+bwd (ms) | Spectral fwd+bwd (ms) |
|---|---|---|---|---|
| 256 | 0.061 | 0.067 | 0.29 | 0.71 |
| 512 | 0.083 | 0.094 | 0.50 | 0.47 |
| 1024 | 0.207 | 0.135 | 0.88 | 0.62 |
| 4096 | 3.23 | 0.35 | 39.7 | 1.50 |

- **SpectralPDE.** Fitted log-log slopes for D ≥ 512 are 1.81 (dense) and 0.63 (spectral). The asymptotic claim holds, but spectral is **slower** than dense up to D = 512 and only 1.5× faster at D = 1024. At the champions' working sizes (D_fast = 256, D_slow = 512) the "32× FLOP reduction" has no wall-clock effect, and the PDE is a negligible share of runtime anyway.
- **SparseHebbian.** Latency against k at D = 1024 is flat (slope 0.02). The implementation scatters the top-k values into a dense vector and calls `torch.outer` on the full D×D. The forward `F.linear(I, W)` is dense too. It is **20–35% slower** than the full Hebbian layer at D ≤ 1024. The "65,536 → 1,024 updates" claim describes non-zero entries, not work done.

![Hebbian scaling](figures/fig5_hebbian_scaling.png)

- **Sequence length.** Cost is linear (slope 0.8–1.0), as expected for a recurrent model. There is no batching across sequences or time anywhere in the training path; batch size is 1 throughout.

### 4.4 Gradient flow

| Model (D = 64 probe) | PDE grad norm (step 10 → 2000) | output_proj grad norm | Ratio |
|---|---|---|---|
| v1 | W 1.3e-5 → 8.8e-5 | 103 → 127 | ~10⁻⁶–10⁻⁷ |
| E20-like | fast 0.12 → 5.0e-5 | 182 → 0.82 | 7e-4 → 6e-5 |
| E34-like | fast 0.17 → 3.2e-3 | 128 → 0.48 | 1e-3 → 7e-3 |

- The dense-PDE "gradient starvation" the paper reports is reproduced, and is worse than reported: 6–7 orders of magnitude below the output projection.
- SpectralPDE starts with about 10⁴× more gradient (the paper reports 215×), but in the E20-like probe it decays back to about 5e-5 by step 2000.
- The cross-scale links end at 1e-7–1e-6, so the slow stream is effectively disconnected.
- Profiler-measured mechanics (§2.1): the resonance phase sits at a stationary point (∂L/∂φ = −2.3e-6 at φ = 0, because $\frac{d}{d\phi}\cos\phi = 0$); Kuramoto `natural_freq` gets exactly 0; `MemoryFormation` detaches its inputs.

---

## 5 Data mining

### 5.1 Codebase census

| Measure | Value |
|---|---|
| Python files / lines | 73 / 12,115 (8,243 source lines), plus `tools/Math-Gen.py` (5,248 lines) |
| Model core (`cellai_core` + v1/v2/v3) | 1,333 source lines |
| Drivers (`arch_search` + `scripts`) | 5,634 source lines |
| Markdown | 30 files, 12,612 lines, ~74k words (docs : code words ≈ 1.75 : 1) |
| Highest complexity | `run_arch_search_v4.main` (cyclomatic 50, rank F) |
| Duplication | each `run_arch_search_vN` shares 9–61% of its lines with the previous version; versions import each other, chaining import-time side effects (§8.1) |
| Tests | 53 pass, 22 fail, 1 collection error; **17% line coverage**; 0% on `arch_search` and `routing` |
| Git | 4 commits by one author. Everything up to E26, including both papers, lands in a single 28k-line commit (`eb18003`) |

Every test failure is API drift, none numeric:
- the registry does not know `v3` (20 tests);
- `MemoryCellBase` was removed;
- `get_system_info` was renamed;
- `CellAIv3` no longer subclasses `CellAIv2`.

### 5.2 The experiment ladder

`data/experiments.csv` lists all 38 rows. Using the papers' own verdicts, 12 of 35 experiments succeeded (34%), 14 failed, 6 were neutral and 3 were partial.

| Round/wave | Success rate |
|---|---|
| R1 (E0–E7) | 2/7 |
| R2 (E8–E14) | 3/7 |
| R3 (E15–E20) | 3/6 |
| R4 (E21–E26) | 1/6 |
| W2 (E27–E29) | 2/3 |
| W3 (E30–E32) | 0/3 |
| W4 (E33–E35) | 1/3 |

![Best-so-far](figures/fig6_best_so_far.png)

Patterns in the table:

1. **Macro NLL hides the plateau.** Macro NLL falls from 19.3 to about 5.4, but nearly all of the late gain is math, which reaches 0.5 nats on templated generator lines. On text+code, the metric the project eventually adopted, the trajectory is flat: E20 7.964 → E22 7.893 → E27 7.847, then eight runs between 7.847 and 7.870, until E34's bypass at 7.300.
2. **Three "experiments" are duplicates.** E9 ≡ E1, E12 ≡ E8, and E5's PPL equals exp of E0's macro NLL, all to 3–4 significant figures. The paper later explains why: the training loop bypassed `cellular_step()`, so resonance never ran ("Bug 1", §13.1). These are reruns of the baseline, not independent observations.
3. **E0–E7 score below uniform guessing.** On text and code they are worse than ln V = 11.5 nats. Initial losses of 110–2,193 nats come from logits $= sE^\top$ with N(0,1) embeddings, so the first few thousand steps are spent recovering from initialisation.
4. **Width barely helps.** D = 256 → 1024 improves text+code by about 0.1 nats (E20/E22 vs E24/E27). This is what to expect if the backbone is not where the predictive information is (§7).
5. **Routing never learned.** All six routing experiments (E4, E11, E17, E19-routing, E20-routing, E31) score 24.7–33.5% on a 3-class task with 33.3% chance.

### 5.3 Claims vs code

| Claim (source) | Finding |
|---|---|
| "125.8 M parameters" (README, paper §3.5, conclusion) | **26.0 M** (§4.1) |
| "E20 … macro PPL 246.6, **966,000×** vs E0" (ASP abstract, §14) | 966,000 = exp(19.3)/246.6, where 19.3 is E0's **final training loss**. E0's held-out macro NLL is 13.415 (ASP §5.1), giving **~2,716×** |
| E20 macro NLL 5.190 (ASP §13.7, §14) | ln(246.6) = **5.508**, the value used in ASP §17 |
| Main paper abstract: PPL "10,000–250,000" | Its own §5.6 reports **450 k–1.18 M** |
| Main paper §5.4: math_v1 and code_v2 both "313.3 → 29.3" | `EVALUATION_REPORT.md` gives math_v1 16.83 → 7.00 and code_v2 313.32 → 29.27; the paper row is copied |
| SpectralPDE "67,600 params, 32× fewer FLOPs" | 65,536 of those parameters are unused; no wall-clock win below D = 1024 |
| SparseHebbian "k² = 1,024 updates", "O(D log D)" | dense D×D outer product; latency independent of k |
| "Online Hebbian adaptation at inference" (main paper §3.4, §6.1) | true for v1/v2. For every v3 champion from E16 on, SparseHebbian is **inactive in `eval()`**, which the v1 paper itself (§5.3) had called a bug |
| v1 "Hebbian always runs; snapshot/restore for fairness" | W is non-negatively incremented on *every* forward, including `chat()`/`benchmark()`. Rank collapse: 99.6% of W's energy in one direction after 2,000 random tokens |
| E30 failed due to "random bypass init" (failure log) | the committed code zero-inits the bypass and sets α₀ = −8, so the stated cause cannot be reproduced from this code |
| "OOD eval (E27): 7.847, no OOD gap" (ASP §20.4) | reshuffles the **same** eval set (`run_arch_search_v7.py:216`); not out of distribution. Same value as E27 to 3 d.p. on a different 90-doc subset |
| E21 and E25 warm macro both 7.229 / 1,379 | identical to 3 d.p. despite different D (256 vs 512) and data order; per-domain values differ, so likely a transcription error |
| Champion of record | README: E20. `eval_protocol_v1.json`: E22/E25. `experiments.yaml`: E27/E29. ASP §20: E34/E29 |
| "Ring Laplacian" over features | `torch.roll(dims=0)` diffuses across 2–8 partitions; for N = 2 it reduces to 2(s_other − s_self) |

---

## 6 Baselines: what the numbers mean

The project never reported a non-neural baseline on its own split. We built one. Records, split permutation and scoring follow the repository exactly:

- **Split A** (drivers v1–v4): used by E21–E26.
- **Split B** (v3-resume and v5–v7): used by E20 and E27–E35.
- Scoring: first 50 eval documents per modality, first 256 tokens, per-document mean NLL averaged over documents.

| Model (trained on the same ~2.2 M train tokens) | Split A text+code NLL | Split B text+code NLL | s.d. over 50-doc samples |
|---|---|---|---|
| Unigram (add-½) | **7.907** | **7.868** | 0.047 |
| Bigram, interpolated Kneser–Ney | 5.522 | 5.458 | 0.10–0.11 |
| Trigram, interpolated Kneser–Ney | 4.991 | 4.895 | 0.13 |

| CellAI run (reported) | Split | text+code NLL | vs unigram on same split |
|---|---|---|---|
| E20 (champion, rounds 1–3) | B | 7.964 | **+0.10 (worse)** |
| E22 (E20 + 8k) | A | 7.893 | −0.01 (within noise; contaminated, §8) |
| E23 / E24 (D = 1024) | A | 8.053 / 7.954 | +0.15 / +0.05 (worse) |
| E25 / E29 (continuous, warm) | A / B | 8.043 / 7.948 | +0.14 / +0.08 (worse) |
| E27 (D = 1024, 16k) | B | 7.847 | −0.02 (within noise; contaminated) |
| E28, E31, E32, E33, E35 | B | 7.859–7.870 | ±0.01 (noise) |
| **E34 (bypass)** | B | **7.300** | −0.57; still **+1.84 vs bigram** |

![Baselines](figures/fig1_baselines.png)

Per modality, CellAI is no better than unigram on text (E22 7.937 vs unigram 8.11) and is **worse** on code (E22 7.848, E24 7.932 vs unigram 7.70; bigram 4.86).

**Interpretation.** With up to 108 M parameters and 16k training steps, the reset-regime champions predict held-out text and code about as well as a token-frequency table. The only run clearly better than unigram is the one with a direct current-token path, and it recovers about a quarter of the gap to a count-based bigram. The 50-document sampling s.d. (~0.05 unigram, ~0.1 bigram) also shows that most Wave 2–4 decisions, made on differences of 0.012–0.023 nats, are inside the noise.

### 6.1 Controlled probe

To separate architecture from data and budget, every model was trained on the same 128 k tokens (the repository's own documentation) at about 6.4 M parameters (D = 64), with the same optimiser, 64-token windows and reset state.

| Model | Val NLL (nats) |
|---|---|
| Unigram counts | 7.958 |
| **Bigram MLP (no state)** | **7.674** |
| Transformer, 2 layers (undertrained at this budget) | 8.028 |
| v1 / v2 / v3-default | 10.69 / 10.66 / 10.69 |
| v3 E20-like (spectral + multiscale) | 7.916 |
| v3 E34-like (+ sparse Hebbian + bypass) | 7.740 |

![Training probe](figures/fig2_training_probe.png)

- The v3 recipes are a large improvement over v1/v2 (10.7 → 7.7–7.9), confirming the *direction* of the E0 → E20 ladder.
- **No Cell variant beats a stateless bigram model.** E20-like barely beats unigram.
- v3 with the default `V3Config()` reproduces v1 bit for bit.
- The transformer result does not show Cell beating transformers: at 128 k tokens and context 64 it is under-trained. The probe shows that the Cell models behave like bigram models.

---

## 7 Why the state carries little: a mechanistic account

**Dense PDE (v1/v2/E0) is input-blind from the zero state.** The update is $dS = \sigma(IW^\top)\odot\tanh(SE^\top) - \gamma S + D\nabla^2 S + \eta\varepsilon$. The input enters **only multiplied by** $\tanh(SE^\top)$, which is exactly 0 at $S = 0$, and every sequence starts from $S = 0$. With noise disabled the state stays exactly zero for any input (measured |S| = 0.0 after 256 random inputs). With the default noise (η = 0.01, dt = 0.01), two completely different 256-token sequences produce aggregate states that differ by **0.35%**; the state is essentially integrated noise.

This explains three things the papers report:
- the "gradient starvation" (∂L/∂W ∝ tanh(SEᵀ) ≈ 0), which was attributed to a zero-initialised gate;
- the persistence of starvation after the gate fix;
- why E0-family models never use context.

**SpectralPDE (v3) is a slow, mostly input-independent integrator.** The update is $S \leftarrow S + 0.01\,(\sigma(\cdot) - 0.05\,S + \dots)$:

- The sigmoid output is always in (0, 1), so each token moves each coordinate by at most 0.01.
- The sigmoid's mean is about 0.5, which produces a large common drift (|S| ≈ 19.7 after 256 tokens).
- The decay time constant is $1/(dt\cdot\gamma) = 2{,}000$ tokens, longer than any 256-token eval document. Evaluation therefore mostly scores the warm-up transient.
- At initialisation, two different input sequences give states differing by **0.2%**.

Training can raise the input filters, but the sigmoid bounds the per-token injection structurally.

**Readout.** The prediction is $(W_H e_t)\odot\sigma(Ga_t+b)$ → $W_o$ → tied logits. The cellular aggregate only gates a linear function of the current token. When the gate carries little information, the best the model can do is a low-rank bigram map through $W_o W_H$ plus the unigram prior in the tied embeddings. Because $W_H$ is driven toward rank 1 by non-negative Hebbian increments, even that map is degraded. E34's bypass adds a clean, separately trainable path $Pe_t$, which is why it is the only intervention that moves text+code NLL.

**v2 extensions are algebraically trivial.** Resonance is $\cos\phi \cdot s$ (a scalar gain, with zero gradient at φ = 0). The lattice is a fixed rank-27 linear map. The Kuramoto term is an input-independent drifting bias with an untrainable frequency. PerFreqResonance (E5/E9/E12/E15/E19/E32) never produced a gain: first because of the bypass bug, then, once fixed, because it duplicates the SpectralPDE's frequency filtering.

---

## 8 Evaluation integrity

### 8.1 Train/eval contamination through import-time seeding

`build_dataset()` shuffles the 18,000 documents with Python's global `random`, then takes the last 15% as eval. Each driver seeds `random` at **import time**, and the drivers import one another. As a result:

- `run_arch_search.py` (seed 0 at import) and v2–v4 resolve to one permutation: **split A**.
- `run_arch_search_v3_resume.py` (which trained the real E20) and v5–v7 resolve to another: **split B**.

We verified this by importing each driver in its natural order and comparing the resulting permutations (`scripts/split_simulation.py`). Combined with checkpoint resumption:

| Run | Trained from | Eval split | Eval docs seen in training by the model's ancestors |
|---|---|---|---|
| E20 | scratch, split B | B | 0% |
| E21, E23–E26 | scratch / E25, split A | A | 0% |
| **E22** | E20 (split B) | A | **85.3%** |
| **E27** | E24 (split A) | B | **85.3%** |
| **E28** | E22 (split A) | B | **85.3%** |
| **E29** | E25 (split A) | B | **85.3%** |
| **E30–E35, H3, "OOD"** | E27 | B | **85.3%** (inherited) |

**Every Wave 2–4 number, including the E34 champion, is scored on documents the starting checkpoint was trained on.** In practice the leak probably flatters these models only slightly, since they barely beat unigram (§6). But it invalidates the comparisons as clean held-out evidence and should be fixed before any further claims are made.

### 8.2 Other protocol issues

- **Test-set model selection.** `TrainHarness` evaluates on `eval_set` during training for E33's early stopping, E34's "in-train peak 6.996" and Hyperband's learning-rate choice (`arch_search/training_v7.py:100-118`). These in-training evals use n = 30, so they are not comparable with the n = 150 headline. E34's headline uses the final model, which mitigates this.
- **"Best" checkpoint.** The `tag="best"` save is called on every eval, so "best" is simply the latest (`training_v7.py:106`).
- **Macro metric.** The project already flags this: math scored on generator-templated lines (no deduplication) reaches ~0.5 nats, and macro PPL improvements are mostly math.
- **Warm continuous eval.** The 4,096-token burn-in stream is built from the eval set itself, so the scored document can appear in its own burn-in.
- **No error bars or seeds.** A single seed, no confidence intervals, and differences of 0.01–0.02 nats are reported as verdicts.
- **Mixed evaluation semantics.** v1/v2 evaluate with online Hebbian updates; v3 SparseHebbian evaluates with W frozen. Cross-generation comparisons therefore mix two different models of "inference".

---

## 9 Engineering findings

| Severity | Finding | Location |
|---|---|---|
| High | Per-token vocabulary head inside the time loop; batching per segment gives a 5–8× training speed-up | `train_step_sequential` (v1/v3) |
| High | In-place `W.data` Hebbian updates on a tensor saved for backward bypass autograd's version check; BPTT silently uses post-update W (the paper's "Hebbian-gradient interference") | `cellai_core/memory.py` |
| High | Train/eval split depends on import order (§8.1) | `arch_search/run_arch_search*.py` |
| Medium | `SpectralPDE.input_gate` unused (65k–1M dead params) | `cellai_core/spectral_pde.py:107` |
| Medium | `freeze_except` does not stop `apply_pending_update`, so E34's "frozen backbone" Phase A still changes W | `arch_search/training_v7.py`, `sparse_hebbian.py` |
| Medium | `MetaplasticityLayer` mutates weights during `chat()`/`benchmark()`/`eval()` | `cellai_core/memory.py:184` |
| Medium | Dense PDE draws `randn` noise during eval, so evaluation is nondeterministic | `cellai_core/base.py` |
| Low | `padding_idx=0` freezes token 0 (`!`) at a zero embedding, so it can never be predicted | encoder |
| Low | Registry, tests and `_profile_baseline.py` out of date; `_test_full.py` hard-requires CuPy | `models/registry.py`, `tests/` |
| Low | `run_round4_followup.py` has a UTF-8 BOM; `run_arch_search_v4.main` has complexity 50 | `arch_search/` |

---

## 10 Discussion

**What the project got right.**
- It identifies real pathologies and largely diagnoses them correctly: dense-PDE starvation, the dead resonance phase, router collapse, Bug 1 (extensions bypassed in training), the cold-start mismatch for continuous trainers, and math inflation of the macro metric.
- It writes these down, keeps a failure log, and moved to a better metric (text+code) mid-programme.
- SpectralPDE + MultiScale is a genuine improvement over the v1 dense design in our probe (10.7 → 7.9 nats at matched size).
- Decoding-time anti-repetition (E13) fixed the degenerate generation loops it targeted.

**What the evidence does not support.**
- That the cellular state contributes useful context. At every scale tested, its contribution to held-out text/code prediction is indistinguishable from zero, given that the reset-regime champions perform at unigram level.
- Claims of efficiency (O(D log D), k² Hebbian) as wall-clock benefits.
- "966,000×", "125.8 M", and online Hebbian adaptation as a property of the v3 champions.
- Wave 2–4 rankings as held-out evidence.

**Why the search plateaued.** The search optimised components that sit behind a gate whose input is almost content-free (§7). Changing D, steps, routing, resonance or document-alignment losses barely moved text+code NLL, because none of them changed how much input information reaches the state. The one intervention that worked, the bypass, routes around the state. That is consistent with the papers' own observation that "cellular diffusion washes out domain-token information" (ASP §13.4). We would put it more strongly: in the current parameterisation, the dynamics barely register input in the first place.

**Paths that could test the core idea fairly.**
1. Make the input enter additively and at full strength (e.g. $S \leftarrow (1-\lambda)S + \lambda\,\phi(W_s S + W_i e_t)$ with λ ~ 0.1–0.5, not dt·σ(·) with dt = 0.01), and remove the multiplicative-at-zero coupling.
2. Read out from the state directly, e.g. LayerNorm(a_t) concatenated with e_t into the head, instead of using the state only as a gate.
3. Add a stateless bigram MLP and a matched GRU/LSTM as permanent baselines; a Cell variant should beat the GRU before any biological component is credited.
4. Ablate the state on the champion (zero $a_t$ at eval). If NLL does not change, the backbone is not contributing.

---

## 11 Recommendations

**Integrity**
1. Build the split once, save the document IDs (`data/splits/arch_eval_v1.json`), and make every driver load it. Re-evaluate E22 and E27–E35 on a clean held-out set.
2. Add the unigram/bigram/trigram baselines (`scripts/ngram_baselines.py`) and a GRU baseline to every results table, with bootstrap 95% CIs and at least 3 seeds for any claimed improvement under 0.1 nats.
3. Use a separate validation split for early stopping and Hyperband; keep the test split untouched.
4. Commit `results_v*.json`, eval manifests and training logs so that reported numbers can be audited.

**Corrections to the write-ups**
5. Parameter count 125.8 M → 26.0 M.
6. 966,000× → ~2,716× (or drop the ratio).
7. E20 macro NLL 5.190 → 5.508.
8. Main-paper abstract PPL range.
9. The duplicated math_v1/code_v2 row.
10. The "OOD" label.
11. The stale champion fields.
12. The online-Hebbian claim for v3.

**Engineering**
13. Batch the vocabulary head per segment (5–8× training speed-up).
14. Remove `input_gate`.
15. Implement a truly sparse Hebbian update (index-based `index_add_` on the k×k block) or drop the complexity claim.
16. Stop mutating weights in `eval()`/`chat()`.
17. Fix the registry and tests.

**Science**
18. Run the state-ablation and input-strength experiments in §10 before further component search.

---

## 12 Conclusion

CellAI is a well-documented, honest research log of an unconventional architecture, and that honesty made this audit possible. Read against the code, against correct baselines, and against a clean split, the evidence says:

- The cellular dynamics, as parameterised, carry almost no information about the input.
- The best models predict held-out text and code at the level of a unigram model. The single meaningful improvement comes from bypassing the cell.
- The efficiency claims do not materialise at the sizes used.
- The most recent rankings rest on a contaminated evaluation split.

None of this rules out biology-inspired sequence models. It means the core hypothesis has not yet been tested: whether reaction–diffusion state with Hebbian plasticity can carry useful context. The changes in §10–§11 would test it, and are cheap.

---

## Appendix A: Reproducing this study

```bash
pip install torch tiktoken pyarrow matplotlib radon pytest
# Split permutations (no data needed; uses dummy labelled records)
python docs/audit/scripts/split_simulation.py run_arch_search_v4 > perm_A.json
python docs/audit/scripts/split_simulation.py run_arch_search_v5 > perm_B.json
# N-gram baselines: needs HF parquet exports of Salesforce/wikitext (wikitext-103-raw-v1/train/0)
# and code-search-net/code_search_net (python/train/0) under $AUDIT_WORK/dl/
AUDIT_WORK=... python docs/audit/scripts/ngram_baselines.py 0 perm_B.json
# Profiling and training probe
AUDIT_OUT=out python docs/audit/scripts/profiling/bench_models.py
AUDIT_OUT=out python docs/audit/scripts/profiling/sweeps.py
AUDIT_OUT=out python docs/audit/scripts/profiling/train_probe.py
```

Note: `ngram_baselines.py` expects the permutation JSON path relative to `$AUDIT_WORK`.

## Appendix B: Files

| File | Content |
|---|---|
| `data/experiments.csv` | 38 rows: id, wave, change, D, steps, params, macro NLL/PPL, text+code NLL, latency, outcome, source section, notes |
| `data/profiling_results.json` | all runtime measurements (params, latency, sweeps, profiler tables, training curves, gradient norms) |
| `data/ngram_split_A_v1-v4.json`, `data/ngram_split_B_v5-v7.json` | baseline NLLs per modality, for 50-doc and full eval sets, with sampling s.d. |
| `figures/fig1–fig7` | baselines, training probe, parameter breakdown, PDE scaling, Hebbian scaling, best-so-far, outcomes by wave |
| `scripts/` | split simulation, n-gram baselines, figure 1, profiling harness |
