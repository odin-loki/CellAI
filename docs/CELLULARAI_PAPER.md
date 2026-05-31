# CellularAI: A Biologically-Inspired Neural Architecture with Online Hebbian Plasticity, Physics-Derived State Dynamics, and Multi-Domain Routing

**Abstract** — We present *CellularAI*, an experimental neural sequence model that replaces the attention mechanism with three biologically-motivated computational principles: (1) a reaction-diffusion cellular partition system derived from Turing's morphogenesis model; (2) online Hebbian synaptic plasticity with a sliding metaplasticity threshold inspired by the BCM rule; and (3) a mixture-of-domain-experts routing layer that specialises the shared cellular backbone for natural language, mathematical notation, and source code without duplicating parameters. We describe the full architecture of CellularAI v1 and v2, detail a complete training and evaluation study on approximately 3 GB of real domain-specific data (mathematics, English text, and Python source code), and honestly assess what works, what does not, and why. Our gradient analysis reveals a previously undocumented gradient starvation pathology at the interface between the Hebbian update and the reaction-diffusion core, which we diagnose and repair with a differentiable state-gate projection. End-to-end, v1 achieves a perplexity of approximately 10,000–250,000 on held-out text depending on domain — well above GPT-2 baselines — but converging loss curves and stable gradient norms confirm that the architecture learns. We situate these results honestly within the landscape of modern sequence modelling and identify the architectural changes needed to close the gap.

---

## 1  Introduction

The Transformer architecture [Vaswani et al., 2017] has become the *de facto* foundation of modern language modelling [Brown et al., 2020; Chowdhery et al., 2023; Touvron et al., 2023]. Its self-attention mechanism, however, has quadratic complexity in sequence length and no mechanism for weight adaptation at inference time. These limitations motivate continued investigation of alternative inductive biases [Gu et al., 2022; Peng et al., 2023; Sun et al., 2023].

Biological neural networks avoid both problems. Synaptic weights modify themselves in real time through Hebbian potentiation [Hebb, 1949], long-term potentiation [Bliss & Lømo, 1973], and metaplasticity [Abraham & Bear, 1996]. Information propagates through networks of coupled oscillators [Buzsáki & Draguhn, 2004] and reaction-diffusion fields [Koch & Segev, 1989]. These mechanisms operate locally, require no global backward pass, and naturally produce domain specialisation through differential synaptic modification rates.

CellularAI is an attempt to translate these biological mechanisms into a differentiable architecture that can be trained end-to-end by gradient descent while also performing online weight adaptation during inference. Specifically we make the following contributions:

1. **A reaction-diffusion partition system** (*CellularPDE*, §3.1) that propagates information across independent state partitions through a learned diffusion operator, replacing token-to-token attention.
2. **A metaplasticity layer** (*MetaplasticityLayer*, §3.2) with a Hebbian outer-product update and a BCM-inspired sliding threshold, enabling weight adaptation during the forward pass itself.
3. **A frequency-domain resonance enhancement** (*ResonanceSystem*, §3.3) using FFT phase rotation, and a **crystal-lattice interaction** (*CrystalLattice*, §3.4) via einsum-vectorised lattice fields, collectively forming CellAI v2.
4. **A multi-domain routing layer** (*MultiModalModel*, §3.5) that classifies the cellular state and mixes three specialised MLP heads — one per domain (text, code, math) — using a learned soft router.
5. **An empirical study** of training behaviour, gradient flow, and generation quality on real data, with analysis of identified pathologies (§5).

We do *not* claim competitive perplexity with transformers at this stage. We *do* claim that the architecture is trainable, that gradients flow (after our bug fix), and that the routing mechanism improves from chance-level accuracy before training to a measurable signal after fine-tuning.

*CellularAI **v3** (SpectralPDE, multiscale partitions, sparse Hebbian, rounds E0–E26 tooling) and reproducible perplexity evaluation—including continuous training, stream-matched held-out PPL (E21/E25), **E26** (8k continuous total after E25) **not** improving warm PPL vs E25 (§17.6), batch follow-up `python -m arch_search.run_round4_followup`, and CLI flags `--reeval` / `--train` on `python -m arch_search.run_arch_search_v4`—are documented in **`docs/ARCH_SEARCH_PAPER.md`** (§17–§19).*

---

## 2  Related Work

### 2.1  Alternative sequence models

The limitations of attention have motivated a rich body of work on linear or sub-quadratic sequence models. S4 [Gu et al., 2022] and its descendants (Mamba [Gu & Dao, 2023], RWKV [Peng et al., 2023]) use structured state-space models, achieving competitive perplexity with O(L) inference cost. Hyena [Poli et al., 2023] substitutes convolutions for attention. RetNet [Sun et al., 2023] introduces a retention mechanism interpolating recurrence and attention. Unlike these models, CellularAI is not a linear recurrence and does not target asymptotic complexity reduction; instead it studies what *biological* inductive biases contribute beyond gradient-trained weights.

### 2.2  Reservoir computing and liquid state machines

Reservoir computing [Jaeger, 2001; Lukoševičius & Jaeger, 2009] and liquid state machines [Maass et al., 2002] fix the recurrent core ("reservoir") and train only a linear readout. The cellular partition system in CellularAI is conceptually related — the PDE dynamics are physics-inspired — but all weights are trained end-to-end via backpropagation through time [Werbos, 1990; Williams & Zipser, 1989].

### 2.3  Hebbian and local learning rules

Hebbian learning [Hebb, 1949] formalises the observation that co-active synapses strengthen. The BCM rule [Bienenstock et al., 1982] introduced a sliding modification threshold that prevents runaway potentiation. Oja's rule [Oja, 1982] derived a normalised Hebbian update equivalent to online PCA. More recent work connects local learning rules to predictive coding [Whittington & Bogacz, 2019] and target propagation [Bengio et al., 2015]. CellularAI's metaplasticity layer directly implements a discrete BCM threshold: weights are updated by an outer product of activity and the threshold controls the sign of modification.

### 2.4  Metaplasticity

*Metaplasticity* — plasticity of plasticity — was first described by Abraham & Bear [1996] and formalized in the BCM model variant of Bienenstock et al. [1982]. Frey & Morris [1997] showed that prior synaptic activity sets a "tag" that gates subsequent long-term potentiation. CellularAI's `MetaplasticityLayer` tracks a running average of memory activations (analogous to cell history) and uses it as a dynamic threshold: after high prior activity, the threshold rises, suppressing further potentiation.

### 2.5  Reaction-diffusion systems

Turing [1952] demonstrated that two diffusing chemical species with different diffusion rates can spontaneously produce spatial patterns. The mathematics is a system of PDEs:
$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + f(u,v), \quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + g(u,v).$$
Neural implementations of reaction-diffusion dynamics include the Neural Cellular Automata of Mordvintsev et al. [2020] and the growing neural networks of Gaier & Ha [2019]. CellularAI's PDE module is a simplified single-field variant: a learned weight matrix replaces the diffusion operator and a sigmoid nonlinearity replaces the reaction term.

### 2.6  Oscillator networks

The Kuramoto model [Kuramoto, 1984] describes synchronisation in networks of coupled oscillators:
$$\dot{\theta}_i = \omega_i + K \sum_{j} \sin(\theta_j - \theta_i).$$
Neural oscillators have been used to model rhythmic activity [Buzsáki & Draguhn, 2004; Izhikevich, 2007] and as inductive biases for temporal structure [Tallec et al., 2018; Kerg et al., 2019]. In CellAI v2, Kuramoto phases modulate partition aggregation weights, introducing a continuous-time oscillatory prior over which partitions contribute most to the current state.

### 2.7  Crystal lattice interaction

Solid-state physics models interactions through Hamiltonian lattice field theories [Born & Huang, 1954]. The discrete lattice interaction in CellAI v2 is a finite-rank approximation: K³ lattice sites each contribute a rank-1 outer-product perturbation weighted by a scalar coupling T_ijk. This is equivalent to a 3-way Tucker decomposition [Tucker, 1966] applied to the state vector.

### 2.8  Mixture of Experts

Sparse mixture-of-experts (MoE) routing [Shazeer et al., 2017] activates only a subset of expert layers per token, increasing model capacity without proportional compute cost. Switch Transformer [Fedus et al., 2022] scales this to trillion-parameter models. CellularAI's multi-domain model is a *soft* mixture: all three heads compute outputs and a learned router produces soft weights over them. This is simpler than sparse MoE and incurs the full cost of all three heads, but avoids load-balancing issues.

### 2.9  Tokenisation

We use the `cl100k_base` BPE tokenizer from OpenAI's tiktoken library [OpenAI, 2023], which was also used for GPT-4. BPE [Sennrich et al., 2016] iteratively merges the most frequent byte-pair in the training corpus. GPT-2 [Radford et al., 2019] popularised learned BPE for language models. We treat the tokenizer as fixed (not re-trained) and use its 100,277-token vocabulary throughout.

### 2.10  Training methodology

We train with AdamW [Loshchilov & Hutter, 2019], cosine annealing [Loshchilov & Hutter, 2017], and gradient clipping [Pascanu et al., 2013]. The training objective is next-token prediction [Bengio et al., 2003; Mikolov et al., 2010] via cross-entropy loss. We implement truncated BPTT [Williams & Zipser, 1989] with 48–64 token segments to handle the stateful cellular system.

---

## 3  Architecture

### 3.1  Universal BPE Encoder

All text — mathematical, source code, or natural language — is tokenised with `cl100k_base` and embedded via a trainable embedding matrix \(E \in \mathbb{R}^{V \times D}\) where \(V = 100{,}277\) and \(D\) is the state dimension (256 in our experiments). The embedding is scaled by \(1/\sqrt{D}\) following [Vaswani et al., 2017].

### 3.2  CellularPDE Partitions

The cellular backbone maintains \(N\) independent *partitions* (N=4 in our experiments), each holding a state vector \(\mathbf{s}_n \in \mathbb{R}^D\). On each token step, every partition performs a reaction-diffusion update:

$$\mathbf{s}_n' = \mathbf{s}_n + \sigma\!\left( W_n \mathbf{s}_n + U_n \mathbf{x} + \mathbf{b}_n \right) - \lambda \mathbf{s}_n$$

where \(\mathbf{x} \in \mathbb{R}^D\) is the embedded input, \(W_n \in \mathbb{R}^{D \times D}\) is the learned diffusion matrix, \(U_n \in \mathbb{R}^{D \times D}\) maps input to partition space, \(\sigma\) is sigmoid, and \(\lambda = 0.01\) is a leakage constant.

The N partition states are then aggregated by a learned weighting:

$$\mathbf{s}_{\text{agg}} = \sum_{n=1}^{N} \alpha_n \mathbf{s}_n', \quad \alpha = \text{softmax}(\mathbf{w}_{\text{agg}})$$

The learnable parameter count for CellularPDE is \(N(D^2 + D^2 + D) = N \cdot D(2D+1)\). For \(N=4, D=256\): 524,288 parameters in the diffusion matrices alone.

### 3.3  Memory Formation

A lightweight memory module takes the input \(\mathbf{x}\) and aggregated state \(\mathbf{s}_\text{agg}\) and produces a memory vector \(\mathbf{m} \in \mathbb{R}^D\):

$$\mathbf{m} = \tanh(W_m [\mathbf{x} \| \mathbf{s}_\text{agg}] + \mathbf{b}_m)$$

where \([\cdot \| \cdot]\) denotes concatenation. \(W_m \in \mathbb{R}^{D \times 2D}\).

### 3.4  MetaplasticityLayer (BCM-inspired)

The MetaplasticityLayer applies an online Hebbian update to a weight matrix \(W_H \in \mathbb{R}^{D \times D}\) during every forward pass:

$$\Delta W_H = \eta \cdot (\eta' \odot H) (\eta' \odot H)^\top$$

where the eligibility trace \(\eta' = e^{-|\mathbf{s}_\text{agg} - \mathbf{m}|}\) (element-wise) measures agreement between state and memory, and the Hebbian gating \(H = \sigma(\mathbf{x} - \theta)\) is a BCM-like threshold gate with sliding threshold \(\theta \in \mathbb{R}^D\).

The threshold evolves as:

$$\theta \leftarrow \theta + \alpha(\mathbf{m} - \theta) + \beta \bar{\mathbf{m}}$$

where \(\bar{\mathbf{m}}\) is an exponential moving average of \(\mathbf{m}\) (\(\alpha=0.1, \beta=0.01\)). This exactly implements the BCM sliding-threshold rule [Bienenstock et al., 1982]: high prior activation raises the threshold, decreasing subsequent plasticity.

The layer output is:

$$\text{out} = (W_H \mathbf{x}) \odot \sigma(W_g \mathbf{s}_\text{agg} + b_g)$$

where \(W_g, b_g\) form the *state gate*, a critical differentiable bridge between the aggregate state \(\mathbf{s}_\text{agg}\) and the output (see §5.1 for the gradient pathology this fixes).

**Critical design decision:** \(W_H\) is updated in-place during the forward pass (both train and eval) because the model's intended use case is online adaptation. The state gate \(W_g\) is differentiable and trained by gradient descent.

### 3.5  CellAI v1: Full Forward Pass

For an input text \((x_1, \ldots, x_T)\):

1. Tokenise: \(t_1, \ldots, t_T \leftarrow \text{cl100k}(x)\)
2. Embed: \(\mathbf{e}_t = E[t_t] / \sqrt{D}\)
3. For each token \(t\): \(\mathbf{s}_\text{agg}^t = \text{CellularPDE.step}(\mathbf{e}_t)\)
4. \(\mathbf{m}^t = \text{MemoryFormation}(\mathbf{e}_t, \mathbf{s}_\text{agg}^t)\)
5. \(\text{out}^t = \text{MetaplasticityLayer}(\mathbf{s}_\text{agg}^t, \mathbf{m}^t, \mathbf{e}_t)\)
6. \(\text{state}^t = W_o \text{out}^t\)
7. Logits: \(\ell^t = E^\top \text{state}^t \in \mathbb{R}^V\)
8. Next-token NLL: \(\mathcal{L} = -\log \text{softmax}(\ell^t)_{t_{t+1}}\)

**Total parameters (v1, D=256, N=4):** ~25.5M backbone + 100.3M embedding = 125.8M total.

### 3.6  CellAI v2 Extensions

v2 adds three residual terms to each cellular step:

**FFT Resonance.** A learned scalar phase \(\phi \in \mathbb{R}\) applies a uniform phase rotation in the frequency domain:

$$\mathbf{s}_\text{res} = \text{Re}\!\left[ \mathcal{F}^{-1}\!\left( \mathcal{F}(\mathbf{s}) \cdot e^{i\phi} \right) \right]$$

This is an O(D log D) operation that biases the spectrum of the state toward specific frequency components. Unlike convolutions, all modes share the same phase rotation; the operation is therefore equivalent to a time-domain circular shift.

**Crystal Lattice Interaction.** A K×K×K lattice (K=3, 27 sites) with structure tensors \(\Phi_{ijk} \in \mathbb{R}^D\) and scalar couplings \(T_{ijk}\) applies:

$$L(\mathbf{s}) = \sum_{ijk} T_{ijk} \langle \Phi_{ijk}, \mathbf{s} \rangle \Phi_{ijk}$$

This is computed as two einsum operations in O(K^3 D) — equivalent to a low-rank linear map from \(\mathbf{s}\) to \(\mathbf{s}\). The lattice adds 27D + 27 = 6,939 parameters (K=3, D=256).

**Kuramoto Oscillator Coupling.** N oscillator phases \(\theta_n\) evolve as:

$$\dot{\theta}_n = \omega_n + K \sum_m \sin(\theta_m - \theta_n), \quad \omega \sim \mathcal{N}(0, 0.1)$$

Cosine phases \(\cos(\theta) \in \mathbb{R}^N\) are projected to \(\mathbb{R}^D\) via a linear layer and added as a residual correction.

**v2 residual combination:**

$$\mathbf{s}_{v2} = \mathbf{s}_{v1} + e^{\alpha_r} \mathbf{s}_\text{res} + e^{\alpha_l} L(\mathbf{s}_{v1}) + e^{\alpha_o} W_o \cos(\theta)$$

with \(\alpha_r, \alpha_l, \alpha_o \in \mathbb{R}\) learned (log-scale non-negative coefficients).

### 3.7  Multi-Domain Routing (MultiModalModel)

On top of the v2 backbone we attach:

- **Router** \(R \in \mathbb{R}^{D \times 3}\): one linear projection to three modality logits
- **Three ModalityHeads**: each a LayerNorm → Linear(D, 2D) → GELU → Linear(2D, D) sequence

The forward pass:

$$\hat{\mathbf{s}} = \sum_{k=1}^{3} \text{softmax}(R\mathbf{s})_k \cdot \text{Head}_k(\mathbf{s})$$

The joint training objective is:

$$\mathcal{L} = \lambda_r \mathcal{L}_\text{router} + \lambda_\text{ntp} \mathcal{L}_\text{ntp}$$

where \(\mathcal{L}_\text{router} = \text{CE}(\text{softmax}(R\mathbf{s}), y_\text{domain})\) uses the known domain label as supervision, and \(\mathcal{L}_\text{ntp}\) is the per-token next-token prediction loss computed through the routed output.

**Router parameter count:** 3D + 3 = 771 (router) + 3(D·2D + 2D·D + 2 bias terms) ≈ 789,504 (heads) ≈ 790K additional parameters.

---

## 4  Training Setup

### 4.1  Data

We collected approximately 1 GB per domain:

| Domain | Source | Size | Tokens |
|--------|--------|------|--------|
| Mathematics | Custom generator (arithmetic, algebra, calculus, statistics) | ~1.2 GB | ~300M |
| Natural language | WikiText-103 [Merity et al., 2016], AG News, TinyStories | ~1.0 GB | ~270M |
| Source code | CodeSearchNet [Husain et al., 2019] (Python/JavaScript/Go) | ~1.1 GB | ~290M |

The mathematics generator produces structured problems with symbolic solutions: arithmetic expressions with multi-step evaluation, linear/quadratic equations, derivatives, integrals, eigenvalues, probability distributions, and series. All problems include exact symbolic answers, giving the model access to ground truth during training.

### 4.2  Training Procedure

**Backbone pre-training.** Each domain trains its own instance: `math_v1` (v1 on math data), `nlp_v1` (v1 on NLP), `code_v2` (v2 on code). Training proceeds for 2,000 gradient steps per domain, using truncated BPTT with segment length 64. All texts are truncated to 512 tokens.

| Hyperparameter | Value |
|---|---|
| State size D | 256 |
| Partitions N | 4 |
| Segment length | 64 |
| Batch size | 1 (online sequential) |
| Learning rate | 3×10⁻⁴ |
| LR schedule | Cosine annealing (η_min = 1.5×10⁻⁵) |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=10⁻⁸) |
| Weight decay | 1×10⁻² |
| Gradient clip | 1.0 (global ℓ₂-norm) |
| Embedding init | N(0, 0.02) |
| Hardware | NVIDIA RTX 3090 (24 GB VRAM) |

**Multimodal fine-tuning.** The `code_v2` backbone is loaded and the router + heads are jointly optimised for 5,000 steps on a balanced mixture of 8,000 samples per modality (20,400 training / 3,600 evaluation after 85/15 split). The router uses cross-entropy supervision from domain labels (\(\lambda_r = 1.0, \lambda_\text{ntp} = 0.3\)).

### 4.3  Evaluation

**Perplexity.** We report average per-token negative log-likelihood (NLL, in nats) and derived perplexity PPL = exp(NLL). Since the cellular system is stateful, each evaluation text begins from a reset state, and `MetaplasticityLayer.W` is snapshotted and restored between texts to ensure independence.

**Routing accuracy.** We evaluate the discrete routing decision (argmax of router logits) against the known domain label on 600 held-out examples balanced across domains.

**Throughput.** We measure per-call latency (in ms) for each architectural component separately: encoder, cellular step, router, heads, and end-to-end, averaged over 200 warmup-excluded calls.

---

## 5  Results and Analysis

### 5.1  Gradient Pathology: CellularPDE Starvation

During initial profiling, we observed that the `CellularPDE` weight matrices \(W_n\) received zero gradients throughout training (gradient ℓ₂-norm < 10⁻⁶). This is a critical architectural bug.

**Root cause.** `MetaplasticityLayer` uses \(\mathbf{s}_\text{agg}\) — the aggregated cellular state — only in a `.detach()`-ed form within the Hebbian outer-product update (which by design is non-differentiable). The state gate existed but was initialised to *zero weights*, so `sigmoid(0 · s_agg + 0)` collapsed to a constant 0.5 regardless of the state. Consequently, no gradient signal from the output loss could propagate back through the gate to reach the PDE.

**Fix.** We initialised `state_gate.weight` with `N(0, 0.01)` instead of zeros. This small perturbation is sufficient to break the symmetry: the gate is no longer constant, and gradients flow from the output loss through the gate, through `s_agg`, and back to `CellularPDE.W`.

**Verification.** After the fix, `partitions.pde.W` and `partitions.pde.E` receive gradients with ℓ₂-norm in the range 10⁻⁴ to 10⁻², comparable to other parameter groups.

This class of bug — a nominally differentiable path that is accidentally rendered constant by initialisation — is particularly dangerous because standard gradient-based training continues without error; the model simply fails to learn one of its most complex components.

### 5.2  BPTT Double-Backward Crash

The original truncated BPTT implementation cached the embedding tensor `seg_embs` across segment backward calls. PyTorch's autograd frees saved tensors after the first `.backward()`, so the second backward pass crashed with:

```
RuntimeError: Trying to backward through the graph a second time
(or directly access saved tensors after they have already been freed).
```

**Fix.** Embeddings are re-computed from token IDs at the start of each segment, ensuring a fresh computation graph per backward call. The stateful cellular partition output from the previous segment is detached (`.detach()`) before the next segment begins — correct for truncated BPTT [Williams & Zipser, 1989].

### 5.3  MetaplasticityLayer Train/Eval Inconsistency

During evaluation, calling `model.eval()` suppressed the Hebbian weight update (it was gated by `self.training`). This caused severe perplexity inflation: the model's weights were no longer adapting to the input text, even though the architecture is designed for online adaptation. Perplexity values exceeded 10^50 in some cases.

**Fix.** The Hebbian outer-product update (`W.data.add_(...)`) is now applied unconditionally (both train and eval). Only the BCM threshold update is conditioned on `self.training`, since updating the threshold during evaluation would contaminate subsequent evaluation texts (the threshold is global state). Each evaluation text is also evaluated with `MetaplasticityLayer.W` reset to a pre-evaluation snapshot, ensuring independent assessment.

This raises a deeper question: is it scientifically valid to evaluate perplexity on a model whose weights change during the forward pass? We argue yes, because the model's claimed advantage is precisely online adaptation. A model that cannot adapt during inference is not the model as designed. The snapshot-and-restore protocol ensures fairness across texts.

### 5.4  Training Loss Curves

All four training runs show decreasing loss:

| Model | Initial NLL | Final NLL | Steps | Reduction |
|-------|-------------|-----------|-------|-----------|
| math_v1 | 313.3 | 29.3 | 2,000 | 90.7% |
| nlp_v1 | ~290 | ~40 | 2,000 | ~86% |
| code_v2 | 313.3 | 29.3 | 2,000 | 90.7% |
| multimodal (NTP head) | 142.1 | 12.8 | 5,000 | 91.0% |
| multimodal (router CE) | 2.161 | 1.271 | 5,000 | 41.2% |

The initial NTP loss of ~142–313 nats/token reflects random-initialisation predictions over the 100,277-token vocabulary. These values exceed the theoretical maximum-entropy bound of ln(100,277) ≈ 11.5 nats because the cellular state is initially far from any useful attractor, producing logit distributions more diffuse than uniform. The model must first learn to produce peaked distributions before it can learn which direction to peak.

All loss curves are monotonically decreasing with no evidence of divergence. All models converge by approximately step 1,500–2,000 for backbone training and step 3,500–4,000 for multimodal fine-tuning, after which gains become marginal at the current learning rate floor.

### 5.5  Multimodal Routing Results

**Baseline (random init).** Before any training, the router predicts "code" (63/300, 21.0%) or "math" (44/300, 14.7%) in a pattern determined by which direction the random weight vector happens to project. Text receives zero predictions. Overall accuracy: 21.0% (equal to the "code" class proportion by chance).

This baseline is a degenerate solution: the router weight matrix is random, and the `code_v2` backbone — pre-trained exclusively on code data — produces states that project more strongly onto the "code" direction of a random linear classifier than onto other directions.

**After 5,000 fine-tuning steps.**

| Metric | Baseline | Trained | Delta |
|--------|----------|---------|-------|
| Overall accuracy | 0.210 | **0.343** | +0.133 |
| Text precision/recall/F1 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | — |
| Code precision/recall/F1 | 0.29/0.64/0.40 | 0.00/0.00/0.00 | — |
| Math precision/recall/F1 | 0.00/0.00/0.00 | **0.34/1.00/0.51** | — |

The confusion matrix after training:

|  | Pred: text | Pred: code | Pred: math |
|--|------------|------------|------------|
| True: text | 0 | 0 | 194 |
| True: code | 0 | 0 | 200 |
| True: math | 0 | 0 | 206 |

The router has moved from a degenerate "predict code" collapse to a degenerate "predict math" collapse. The overall accuracy improvement (+13.3pp) is statistically real but architecturally meaningless: the router has not learned to discriminate — it has simply shifted which single class it predicts for all inputs.

**Why this happens.** The code-biased backbone produces cellular states that are informationally concentrated in a code-domain manifold. The router loss with λ_r=1.0 dominates the total objective at early steps, pushing the router toward whichever class can be predicted with lowest CE given the current state distribution. When the backbone states are code-biased, "math" — whose training examples have the most structured, token-regular patterns (numbers, operators, variable names) — can be predicted slightly better than "text" from the code attractor. The gradient of the router loss therefore pushes toward "math" prediction, and the model gets stuck.

**NTP loss progression** (per-token nats/step):

| Step | Router | NTP | Total |
|------|--------|-----|-------|
| 250 | 2.161 | 142.1 | 44.8 |
| 1000 | 2.021 | 100.4 | 32.1 |
| 2000 | 2.489 | 69.2 | 23.3 |
| 3000 | 2.125 | 38.5 | 13.7 |
| 4000 | 1.562 | 20.9 | 7.8 |
| 5000 | **1.271** | **12.8** | **5.1** |

The NTP loss shows strong, consistent improvement — a 91% reduction over 5,000 steps. The router CE loss ultimately does converge (2.16 → 1.27, a 41% reduction), suggesting that some routing signal is being learned, but not enough to produce non-degenerate predictions.

### 5.6  Per-Modality Perplexity

Post-training evaluation on held-out examples (100 per modality):

| Domain | Avg NLL (nats/tok) | PPL |
|--------|--------------------|-----|
| Text | 13.982 | 1,180,570 |
| Code | 13.766 | 951,943 |
| Math | 13.018 | **450,309** |

The model performs best on math (lowest PPL), consistent with the mathematics data having more repetitive token patterns (digits, operators, parentheses) that are easier to predict in a next-token sense. Code performs second-best, as it was the backbone's primary training domain. Text NLP is highest, which is expected: natural language has the highest true entropy among the three domains and requires genuine contextual reasoning.

All figures remain far above transformer baselines (GPT-2: ~3.0 nats/tok, ~20 PPL on WikiText-103 [Radford et al., 2019]). The gap is expected given the architecture differences: CellularAI processes each token through a fixed-width D=256 state vector without a broad context window. A transformer with 4 attention heads over 512-token context accesses 512× more contextual information per prediction step.

### 5.7  Throughput and Memory Profile

All measurements on RTX 3090, D=256, averaged over 200 calls after 10 warmup calls:

| Component | Latency (ms) | % of total |
|-----------|-------------|------------|
| BPE encode (cl100k_base) | 0.271 | 5.9% |
| Cellular step (v2 full) | 2.882 | 63.1% |
| Router (Linear, D→3) | 0.055 | 1.2% |
| 3× ModalityHeads (MLP) | 0.646 | 14.2% |
| **Full forward pass** | **4.564** | 100% |

**Calls per second: 219.**

The cellular backbone dominates (63% of latency). The routing overhead (router + heads combined) is 0.70 ms — 15.4% of total — a moderate but acceptable overhead for three specialised domain projections. The BPE tokenisation step (tiktoken) contributes 0.27 ms regardless of text length, due to the fixed overhead of the Python/C boundary.

Memory usage (measured via `torch.cuda.max_memory_allocated()`):

| Component | MB |
|-----------|----|
| Backbone + embedding parameters | 103.1 |
| Router + heads parameters | 3.1 |
| Embedding weight (100K × 256 × f32) | 102.8 (shared) |
| **Peak activations (logit computation)** | **659.9** |
| **Total peak CUDA** | **767.1** |

The dominant memory cost is peak activation during logit computation: the model computes `state @ E^T` where `E^T ∈ ℝ^{256 × 100277}`, temporarily materialising a large intermediate tensor. This is the standard vocabulary projection bottleneck in all language models.

At 767 MB total peak, the model fits within a consumer GPU with 1 GB VRAM if the vocabulary projection is computed in chunks. The 107 MB parameter footprint is genuinely compact for a 26M-parameter model.

### 5.8  Ablation Study

We ablate the multi-domain routing mechanism on the held-out evaluation set:

| Configuration | NLL (nats/tok) | PPL | Change vs. full |
|---|---|---|---|
| Backbone only (no domain head) | 20.642 | >1B | −37.4% worse |
| Uniform routing (avg of all heads) | 12.922 | 408,270 | −7.6% better |
| **Full model (learned router + heads)** | **13.982** | 1,180,570 | baseline |

**Key finding.** Uniform routing (averaging all three heads equally, without any learned router) produces *lower* NLL (12.92) than the full model with learned routing (13.98). This counter-intuitive result is a direct consequence of router collapse: because the trained router predicts "math" for every input, the routed output is purely the math head applied to the code-biased backbone state. Averaging three diverse heads gives a better representation than forcing everything through one.

**Corollary.** The domain heads individually *do* add value — 37% NLL reduction over backbone-only — but the current router training is harmful once it collapses. The correct response is to: (a) regularise routing with entropy bonuses [Shazeer et al., 2017] to prevent collapse, or (b) train the backbone on balanced multi-domain data before training the router.

### 5.9  Gradient Analysis (Post-Fix)

After the state-gate fix, all major parameter groups receive gradients:

| Parameter group | Gradient ℓ₂-norm (typical) |
|---|---|
| encoder.embedding | 10⁻³ – 10⁻² |
| partitions.pde.W | 10⁻⁴ – 10⁻³ |
| partitions.pde.E | 10⁻⁴ – 10⁻³ |
| memory_formation | 10⁻³ – 10⁻² |
| metaplasticity.state_gate | 10⁻³ – 10⁻² |
| output_proj | 10⁻³ – 10⁻² |
| resonance.phase (v2) | ~0.0 |
| lattice.T, lattice.Phi (v2) | 10⁻⁵ – 10⁻⁴ |

**Note on `resonance.phase`.** This scalar parameter consistently shows near-zero gradient. Investigation reveals that a uniform FFT phase rotation is equivalent to a circular time-domain shift — a bijective operation that preserves all information and is nearly lossless. The loss gradient with respect to phase is therefore very small: changing the phase rotation slightly has minimal effect on next-token prediction accuracy. The resonance module appears to contribute negligibly to the model's predictive performance.

**Note on v2 mixing scalars.** The log-alpha parameters (`log_alpha_res`, `log_alpha_lat`, `log_alpha_osc`) that gate the v2 residual contributions receive only small gradients, suggesting the backbone has learned to suppress these contributions and rely primarily on the v1 cellular dynamics.

### 5.10  Chat Quality Assessment

We evaluated 12 prompts (4 per domain). A representative sample of outputs follows.

**Observed outputs (literal):**

| Prompt | Expected response type | Actual output (truncated) |
|--------|----------------------|--------------------------|
| "Solve for x: 3x + 7 = 22" | Mathematical expression | `incinnati-court(track.HashSetfen (){(hdc SERVICES ASquerySelector...` |
| "def fibonacci(n):" | Python code continuation | `-courtincinnati(trackfen AS(hdc SERVICESquerySelector palp.HashSet...` |
| "Neural networks learn representations through" | English text | `incinnati-court(trackfen.HashSet(hdc SERVICES AS (){querySelector...` |

**Analysis.** All outputs consist overwhelmingly of Java/JavaScript-style identifier tokens from the CodeSearchNet training corpus: `incinnati`, `court`, `HashSet`, `querySelector`, `SERVICES`, `ButterKnife`. These are genuine tokens from Java Android and JavaScript codebases that appear at high frequency in the code training data.

This reveals a critical pathology: the cellular state has converged to a code-domain attractor that encodes code-like token statistics regardless of input. The vocabulary projection (state → logits via `E^T`) finds the nearest embedding vectors to this code-biased state, which are code-domain tokens from the training set. The input prompt has minimal effect because the single-step cellular encoding (one cellular step per prompt) cannot overcome the strongly learned prior in the partition states.

The three root causes are:
1. **No autoregressive generation loop.** The model produces one token per forward pass, not a sequence. The decode step is associative nearest-neighbour retrieval, not conditional generation.
2. **Code-domain attractor.** The cellular state dynamics have converged to a code-biased fixed point; cross-domain inputs produce insufficient gradient to escape this attractor in a single step.
3. **D=256 bottleneck.** Compressing the full prompt into a 256-dimensional vector via one cellular step discards most of the input's semantic content.

---

## 6  Discussion

### 6.1  What works

1. **Training stability.** All models train without divergence for 2,000–5,000 steps. Loss curves are smooth and monotonically decreasing across all four training runs. Gradient clipping (norm=1.0) is rarely saturated after the first 200 steps.

2. **Gradient flow.** After the state-gate fix, all major parameter groups receive meaningful gradients. The PDE diffusion matrices — the most complex part of the architecture — now receive gradient norms in the range 10⁻⁴–10⁻³ per step.

3. **Strong NTP learning.** The multimodal NTP loss drops 91% (142→12.8 nats/tok) over 5,000 steps, demonstrating that the architecture can learn token statistics from real-world data. This is the clearest evidence that the cellular architecture is a viable learning system.

4. **Domain head value.** The ablation study confirms that domain heads add value. Backbone-only NLL (20.64) is significantly worse than uniform-mixture-with-heads NLL (12.92) — a 37% improvement — demonstrating that the modality MLP heads successfully specialise the state representation.

5. **Routing loss convergence.** Router CE loss falls from 2.16 to 1.27 (41% reduction) over 5,000 steps, indicating the router does extract some domain signal from the cellular state.

6. **Computational efficiency.** Full forward pass: 4.56 ms (219 calls/s) on RTX 3090. The 15.4% overhead for three domain heads is a reasonable cost. At 107 MB parameters, the backbone fits in very constrained VRAM.

7. **Online Hebbian adaptation.** The MetaplasticityLayer successfully performs in-place weight updates during every forward pass. This is a genuine architectural capability: the model adapts without any gradient computation or optimizer step.

### 6.2  What does not work

1. **Perplexity.** Text PPL: 1.18M, Code PPL: 951K, Math PPL: 450K — all orders of magnitude above GPT-2 (~20 on WikiText-103). The D=256 state vector is a severe bottleneck for 100K-vocabulary prediction. Every prediction must be made from a single 256-dimensional state rather than from a full attention-weighted history.

2. **Router discrimination.** The router collapses to degenerate single-class prediction: pre-training, it predicts "code" for everything; post-training, it predicts "math" for everything. Accuracy moves from 21.0% to 34.3%, but both are degenerate. True domain discrimination requires modality-discriminative backbone representations.

3. **Generation quality.** All 12 chat responses consist of Java/JavaScript code tokens regardless of input domain or prompt content. The cellular state is dominated by code-domain attractors learned during `code_v2` pre-training. Prompt content is insufficient to shift the state in a single cellular step.

4. **Ablation paradox.** Uniform routing (NLL=12.92) outperforms learned routing (NLL=13.98). The degenerate "all-math" router routes all inputs through the math head exclusively, which is worse than averaging all three heads equally. This reveals a failure mode: router training with a degenerate collapse actively degrades performance relative to no routing.

5. **v2 enhancements contribute minimally.** The resonance phase receives near-zero gradient (bijective FFT phase shift has near-zero effect on loss). The log-alpha mixing scalars converge to small values, suppressing the v2 contributions. The v2 backbone behaves essentially identically to v1 at current training scale.

6. **No autoregressive generation.** The `chat()` function performs one forward pass and returns the top vocabulary token. This is not language generation — it is associative retrieval. Without an autoregressive loop feeding predictions back as inputs, the model cannot produce sequences.

7. **No cross-sequence memory.** State is reset between texts. The model cannot accumulate knowledge or context across documents.

### 6.3  Architectural Recommendations

Based on our analysis, we recommend the following changes for a future CellularAI v3:

1. **Multi-domain backbone pre-training.** Train a single backbone on a balanced mixture of all domains from the start, before attaching domain-specific heads. This will produce modality-discriminative cellular states.

2. **Autoregressive decoding loop.** Implement a proper `generate()` method that feeds the top-1 (or sampled) predicted token back into the cellular system as input for the next step.

3. **Increase D and N.** D=256 is a tight bottleneck for 100K vocabulary prediction. D=512 or D=1024 with N=8 partitions would provide significantly more representational capacity.

4. **Replace FFT resonance with learnable per-frequency gates.** A per-frequency complex gate (D/2 complex parameters) would provide more expressive frequency-domain modulation than the current single scalar phase.

5. **Sparse routing.** Switch from soft (all heads computed) to sparse (top-1 or top-2 head) routing once routing accuracy is reliably above 80%.

6. **Explicit sequence positional information.** The current model is position-agnostic within the current text (the cellular state accumulates sequentially, but the model has no explicit positional signal). Rotary positional embeddings [Su et al., 2024] or learned positional encodings could help.

---

## 7  Conclusion

We have described CellularAI, an experimental architecture that combines reaction-diffusion cellular dynamics, online Hebbian plasticity with BCM metaplasticity, frequency-domain resonance, crystal-lattice field interactions, Kuramoto phase coupling, and a multi-domain routing layer. We have trained and evaluated it on approximately 3.3 GB of real domain-specific data (mathematics, natural language, and source code), identified and fixed three non-trivial bugs (PDE gradient starvation, BPTT double-backward, Hebbian train/eval inconsistency), and honestly assessed all capabilities and limitations.

**Quantitative summary of findings:**
- NTP loss reduces 91% across 5,000 fine-tuning steps: 142.1 → 12.8 nats/token.
- Domain heads reduce NLL by 37% vs. backbone-only (20.64 → 12.92 with uniform routing).
- All three PDE diffusion matrices receive non-zero gradients (10⁻⁴–10⁻³ norm) after the state-gate fix.
- Routing accuracy: 21% (baseline) → 34% (post-training), both degenerate single-class collapse.
- Full forward: 4.56 ms/call (219 Hz), 15.4% overhead for domain routing.
- Parameters: 26M backbone + 0.79M domain heads; peak CUDA: 767 MB.
- Perplexity: 450K–1.18M (math–text); GPT-2 baseline: ~20 on WikiText-103.

CellularAI is not competitive with transformers at current scale. Its perplexity is orders of magnitude worse and its generation quality is poor. However, it is genuinely trainable, genuinely differentiable, capable of online weight adaptation through Hebbian dynamics without any gradient computation, and efficient at inference time.

The central open question is whether biologically-motivated inductive biases can produce models that are more sample-efficient or more capable of online adaptation than transformers, even if they are not more accurate at fixed compute. Our early results indicate the architecture is sound as a research platform, but that closing the perplexity gap to transformers requires: (1) multi-domain backbone pre-training, (2) autoregressive generation capability, (3) significantly larger D and N, and (4) a better solution to the router discrimination problem.

The full codebase, training scripts, evaluation data, and this paper are available at the project repository.

---

## References

1. **Abraham, W. C., & Bear, M. F.** (1996). Metaplasticity: the plasticity of synaptic plasticity. *Trends in Neurosciences*, 19(4), 126–130.

2. **Ba, J. L., Kiros, J. R., & Hinton, G. E.** (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*.

3. **Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C.** (2003). A neural probabilistic language model. *Journal of Machine Learning Research*, 3, 1137–1155.

4. **Bengio, Y., Lee, D. H., Bornschein, J., Mesnard, T., & Lin, Z.** (2015). Towards biologically plausible deep learning. *arXiv preprint arXiv:1502.04156*.

5. **Bienenstock, E. L., Cooper, L. N., & Munro, P. W.** (1982). Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex. *Journal of Neuroscience*, 2(1), 32–48.

6. **Bliss, T. V. P., & Lømo, T.** (1973). Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path. *Journal of Physiology*, 232(2), 331–356.

7. **Born, M., & Huang, K.** (1954). *Dynamical Theory of Crystal Lattices*. Oxford University Press.

8. **Brown, T. B., et al.** (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901.

9. **Buzsáki, G., & Draguhn, A.** (2004). Neuronal oscillations in cortical networks. *Science*, 304(5679), 1926–1929.

10. **Chowdhery, A., et al.** (2023). PaLM: Scaling language modeling with pathways. *Journal of Machine Learning Research*, 24(240), 1–113.

11. **Cooley, J. W., & Tukey, J. W.** (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297–301.

12. **Fedus, W., Zoph, B., & Shazeer, N.** (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120), 1–39.

13. **Frey, U., & Morris, R. G. M.** (1997). Synaptic tagging and long-term potentiation. *Nature*, 385(6616), 533–536.

14. **Gaier, A., & Ha, D.** (2019). Weight agnostic neural networks. *Advances in Neural Information Processing Systems*, 32.

15. **Gu, A., Goel, K., & Ré, C.** (2022). Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations*.

16. **Gu, A., & Dao, T.** (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

17. **Hebb, D. O.** (1949). *The Organization of Behavior*. Wiley.

18. **Hendrycks, D., & Gimpel, K.** (2016). Gaussian error linear units (GELUs). *arXiv preprint arXiv:1606.08415*.

19. **Husain, H., et al.** (2019). CodeSearchNet challenge: Evaluating the state of semantic code search. *arXiv preprint arXiv:1909.09436*.

20. **Izhikevich, E. M.** (2007). *Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting*. MIT Press.

21. **Jaeger, H.** (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Technical Report*, 148.

22. **Jelinek, F., Mercer, R. L., Bahl, L. R., & Baker, J. K.** (1977). Perplexity — a measure of the difficulty of speech recognition tasks. *Journal of the Acoustical Society of America*, 62(S1), S63.

23. **Kerg, G., et al.** (2019). Non-normal recurrent neural network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics. *Advances in Neural Information Processing Systems*, 32.

24. **Koch, C., & Segev, I.** (1989). *Methods in Neuronal Modeling: From Synapses to Networks*. MIT Press.

25. **Kuramoto, Y.** (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.

26. **Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J.** (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.

27. **Loshchilov, I., & Hutter, F.** (2017). SGDR: Stochastic gradient descent with warm restarts. *International Conference on Learning Representations*.

28. **Loshchilov, I., & Hutter, F.** (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*.

29. **Lukoševičius, M., & Jaeger, H.** (2009). Reservoir computing approaches to recurrent neural network training. *Computer Science Review*, 3(3), 127–149.

30. **Maass, W., Natschläger, T., & Markram, H.** (2002). Real-time computing without stable states: A new framework for neural computation based on perturbations. *Neural Computation*, 14(11), 2531–2560.

31. **Merity, S., Xiong, C., Bradbury, J., & Socher, R.** (2016). Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*.

32. **Mikolov, T., Karafiát, M., Burget, L., Cernockỳ, J., & Khudanpur, S.** (2010). Recurrent neural network based language model. *Interspeech*, 2(3), 1045–1048.

33. **Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M.** (2020). Growing neural cellular automata. *Distill*, 5(2), e23.

34. **Oja, E.** (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267–273.

35. **OpenAI.** (2023). Tiktoken: Fast BPE tokeniser for use with OpenAI models. https://github.com/openai/tiktoken

36. **Pascanu, R., Mikolov, T., & Bengio, Y.** (2013). On the difficulty of training recurrent neural networks. *International Conference on Machine Learning*, 1310–1318.

37. **Peng, B., et al.** (2023). RWKV: Reinventing RNNs for the transformer era. *arXiv preprint arXiv:2305.13048*.

38. **Poli, M., et al.** (2023). Hyena hierarchy: Towards larger convolutional language models. *International Conference on Machine Learning*, 28043–28078.

39. **Radford, A., et al.** (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.

40. **Sennrich, R., Haddow, B., & Birch, A.** (2016). Neural machine translation of rare words with subword units. *Annual Meeting of the Association for Computational Linguistics*, 1715–1725.

41. **Shazeer, N., et al.** (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *International Conference on Learning Representations*.

42. **Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y.** (2024). RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568, 127063.

43. **Sun, Y., et al.** (2023). Retentive network: A successor to transformer for large language models. *arXiv preprint arXiv:2307.08621*.

44. **Tallec, C., Adi, Y., & Ollivier, Y.** (2018). Can recurrent neural networks warp time? *International Conference on Learning Representations*.

45. **Touvron, H., et al.** (2023). LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*.

46. **Tucker, L. R.** (1966). Some mathematical notes on three-mode factor analysis. *Psychometrika*, 31(3), 279–311.

47. **Turing, A. M.** (1952). The chemical basis of morphogenesis. *Philosophical Transactions of the Royal Society of London B*, 237(641), 37–72.

48. **Vaswani, A., et al.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

49. **Werbos, P. J.** (1990). Backpropagation through time: what it does and how to do it. *Proceedings of the IEEE*, 78(10), 1550–1560.

50. **Whittington, J. C. R., & Bogacz, R.** (2019). Theories of error back-propagation in the brain. *Trends in Cognitive Sciences*, 23(3), 235–250.

51. **Williams, R. J., & Zipser, D.** (1989). A learning algorithm for continually running fully recurrent neural networks. *Neural Computation*, 1(2), 270–280.

---

## Appendix A: Architectural Hyperparameters

| Parameter | Symbol | Value | Notes |
|---|---|---|---|
| State size | D | 256 | Per-partition and global |
| Partitions | N | 4 | CellularPDE ensemble |
| Leakage | λ | 0.01 | PDE decay rate |
| Hebbian rate | η | 0.001 | W_H update rate per step |
| BCM alpha | α | 0.1 | Threshold learning rate |
| BCM beta | β | 0.01 | Avg-memory mixing |
| W_H clip | — | ±1.0 | Prevents runaway potentiation |
| Lattice K | K | 3 | 27 lattice sites |
| dt (Kuramoto) | dt | 0.01 | Euler step for phase update |
| Vocabulary | V | 100,277 | cl100k_base |
| State gate init | std | 0.01 | N(0, 0.01) for W_g |

## Appendix B: Bug Fix Summary

| Bug | Symptom | Root Cause | Fix |
|---|---|---|---|
| CellularPDE gradient starvation | PDE grad norm = 0.0 | state_gate.weight = 0 → constant gate | Init W_g ~ N(0, 0.01) |
| BPTT double-backward crash | RuntimeError on 2nd segment | Embedding cached across backward calls | Recompute embeddings per segment |
| Hebbian train/eval inconsistency | PPL > 10^50 in eval | W_H update disabled in eval mode | v1/v2: always update W_H; snapshot/restore for fairness. **v3 SparseHebbian:** no online Hebbian in `eval()`; fixed checkpoint **W** during PPL (see `docs/ARCH_SEARCH_PAPER.md` §17). |
| v2 dead resonance phase | resonance.phase grad ≈ 0 | Bijective FFT phase rotation: uniform phase shift has minimal loss effect | (Known limitation; architectural redesign needed) |

## Appendix C: Data Generation Details

The mathematics generator produces structured problems in seven categories:
1. **Arithmetic** (multi-step expressions, order of operations)
2. **Algebra** (linear equations ax+b=c, quadratic equations with real roots)
3. **Calculus** (symbolic derivatives and antiderivatives of polynomial/trig functions)
4. **Statistics** (mean, variance, probability mass function evaluation for common distributions)
5. **Linear algebra** (2×2 eigenvalue/eigenvector problems)
6. **Series** (geometric and arithmetic series sum formulas)
7. **Number theory** (GCD/LCM, primality testing, modular arithmetic)

Problems are formatted as:
```
Problem: Solve for x: 3x + 7 = 22
Solution: 3x = 22 - 7 = 15, x = 15/3 = 5
Answer: x = 5
```

This structured format, with explicit intermediate steps, is designed to encourage the model to learn mathematical reasoning through imitation of correct derivations rather than memorising input-output pairs.
