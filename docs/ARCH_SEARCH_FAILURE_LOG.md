# Architecture search — failure log

Negative results are first-class science. Brief bullets for post-mortems and paper §20.

---

## Wave 3

| Exp | Hypothesis | Outcome | Plausible cause |
|-----|------------|---------|-----------------|
| **E30** | Encoder→state bypass preserves modality | **Catastrophic** (text+code NLL 133.6) | Random bypass init + simultaneous backbone fine-tune destabilised PDE/Hebbian equilibrium |
| **E31** | Input-side PDE banks improve routing | **Neutral** NLL; routing **24.7%** | Banks add capacity but router from raw embedding doesn't discriminate; diffusion still dominates |
| **E32** | PerFreqResonance at D=1024 | **Neutral** (+0.023 NLL) | SpectralPDE already captures frequency structure; gated resonance redundant |

## Round 4 (historical)

| Exp | Outcome | Note |
|-----|---------|------|
| **E26** | Continuous warm PPL **regressed** vs E25 | Resume + same LR without EMA/held-out early-stop |
| **E5/E9** | PerFreq at D=256 neutral/negative | Led to E32 retest at scale — still neutral |

## Wave 2 (non-failures documented for discipline)

| Exp | Outcome | Note |
|-----|---------|------|
| **E28** | +0.016 NLL vs E27 | Longer D=512 training cannot beat D=1024 champion |

## Wave 4

| Exp | Hypothesis | Outcome | Plausible cause |
|-----|------------|---------|-----------------|
| **E33** | E27 +16k (32k total) | **No beat** (7.859; best mid-train 7.642) | Early-stop at 3.2k; longer D=1024 fine-tune does not improve over E27 |
| **E35** | Token-state doc-align aux | **No beat** (7.864) | Auxiliary cosine loss does not transfer to headline CE |
| **Hyperband** | LR screen 600×3 from E27 | H3 best 7.646 | Short screens; no full-run promotion |

**E34 (bypass retry)** — **success**: zero-init bypass + freeze 1k / unfreeze 3k → text+code NLL **7.300** (in-train best **6.996** @ step 3000). Beats E27 (**7.847**). Staged unfreeze fixes E30 catastrophic init.

**Diagnostics:** OOD text+code NLL **7.847** (matches in-distribution E27). Decode sweep diversity 0.73–1.0 across rep/noise grid.
