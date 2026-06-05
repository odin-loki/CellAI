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

---

*Update when Wave 4 (E33–E35) completes.*
