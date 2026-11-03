from common import *
import glob, math, random
import torch.nn as nn, torch.nn.functional as F
import tiktoken
name = sys.argv[1]; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
SEG = 64; LR = 1e-3; EVAL_EVERY = 100
enc = tiktoken.get_encoding("cl100k_base")
import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# The 26 original docs (exclude this audit folder).
files = sorted(f for f in glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True)
               if os.sep + "audit" + os.sep not in f)
toks = []
for f in files: toks += enc.encode(open(f, encoding="utf-8").read(), disallowed_special=())
toks = torch.tensor(toks); n = len(toks); split = int(0.9*n)
train, val = toks[:split], toks[split:]
V = enc.n_vocab
rng = random.Random(0)
val_starts = list(range(0, len(val)-SEG-1, (len(val)-SEG-1)//24))[:24]

class TinyTransformer(nn.Module):
    def __init__(self, D=64, L=2, H=4, ctx=SEG):
        super().__init__()
        self.emb = nn.Embedding(V, D); self.pos = nn.Embedding(ctx, D)
        layer = nn.TransformerEncoderLayer(D, H, 4*D, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, L); self.ln = nn.LayerNorm(D)
    def forward(self, ids):
        T = ids.shape[-1]
        h = self.emb(ids) + self.pos(torch.arange(T))
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        return self.ln(self.blocks(h, mask=mask, is_causal=True)) @ self.emb.weight.t()

def cell_build(name):
    small = ModelParams(device="cpu", state_size=64, num_partitions=2)
    if name == "v1": return CellAI(small)
    if name == "v2": return CellAIv2(small)
    base = dict(pde_type="spectral", partition_type="multiscale", N_fast=2, D_fast=32, N_slow=2, D_slow=64, K_slow=8)
    if name == "v3_E20like": return CellAIv3(small, V3Config(label=name, hebbian_type="sparse", resonance_type="none", **base))
    if name == "v3_E14like": return CellAIv3(small, V3Config(label=name, hebbian_type="full", resonance_type="per_freq", **base))
    if name == "v3_E34like": return CellAIv3(small, V3Config(label=name, hebbian_type="sparse", use_bypass=True, **base))
    if name == "v3_dense_single": return CellAIv3(small, V3Config(label=name))
    raise KeyError(name)

class Bigram(nn.Module):
    """Tied-embedding bigram: logits_t = W2 tanh(W1 e(x_t)) @ E^T  -- no recurrence, no state."""
    def __init__(self, D=64):
        super().__init__(); self.emb = nn.Embedding(V, D); self.f = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, D))
    def forward(self, ids): return self.f(self.emb(ids)) @ self.emb.weight.t()
is_tf = name in ("transformer", "bigram")
m = TinyTransformer() if name == "transformer" else Bigram() if name == "bigram" else cell_build(name)
nparam = sum(p.numel() for p in m.parameters())

def seq_logits(ids):
    """ids (T+1,) -> logits (T, V) for predicting ids[1:]; reset state (reset regime)."""
    if is_tf: return m(ids[:-1].unsqueeze(0))[0]
    m.partitions.reset()
    if hasattr(m.memory_formation, "reset"): m.memory_formation.reset()
    embs = m.encoder.embedding(ids) * m.encoder._scale
    S = torch.stack([m.cellular_step(embs[t]) for t in range(len(ids)-1)])
    return S @ m.encoder.embedding.weight.t()

def evaluate():
    was = m.training
    # Note: Cell v1/v2 MetaplasticityLayer mutates W even in eval; keep train() to mimic repo eval and snapshot W
    snap = {k: v.clone() for k, v in m.state_dict().items()}
    tot = 0.0; cnt = 0
    with torch.no_grad():
        for s in val_starts:
            ids = val[s:s+SEG+1]
            tot += F.cross_entropy(seq_logits(ids), ids[1:], reduction="sum").item(); cnt += SEG
    m.load_state_dict(snap); m.train(was)
    return tot/cnt

opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR*0.05)
log = {"name": name, "params": nparam, "train_tokens_total": len(train), "val_tokens_total": len(val),
       "steps": STEPS, "seg": SEG, "train": [], "val": [], "grad_norms": []}
m.train(); t0 = time.perf_counter(); run = 0.0
log["val"].append({"step": 0, "nll": evaluate()})
for step in range(1, STEPS+1):
    s = rng.randrange(0, len(train)-SEG-1); ids = train[s:s+SEG+1]
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(seq_logits(ids), ids[1:])
    loss.backward()
    if not is_tf and hasattr(m.metaplasticity, "apply_pending_update"): m.metaplasticity.apply_pending_update()
    if step in (1, 10, 100, STEPS):
        g = {}
        for n_, p in m.named_parameters():
            key = ".".join(n_.split(".")[:3]) if n_.startswith("partitions") else ".".join(n_.split(".")[:2])
            gn = 0.0 if p.grad is None else float(p.grad.norm())**2
            g[key] = g.get(key, 0.0) + gn
        log["grad_norms"].append({"step": step, "norms": {k: math.sqrt(v) for k, v in g.items()}})
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); sched.step(); run += loss.item()
    if step % 25 == 0:
        log["train"].append({"step": step, "nll": run/25}); run = 0.0
    if step % EVAL_EVERY == 0 or step == 50:
        v = evaluate(); log["val"].append({"step": step, "nll": v})
        print(f"{name} step {step} train {log['train'][-1]['nll']:.3f} val {v:.3f} ppl {math.exp(v):.0f} t={time.perf_counter()-t0:.0f}s", flush=True)
log["wall_s"] = time.perf_counter()-t0
log["tokens_per_s"] = STEPS*SEG/log["wall_s"]
# unigram baseline on val (add-1 smoothing over train counts)
if name in ("transformer", "bigram"):
    cnt = torch.bincount(train, minlength=V).float() + 1; p = cnt/cnt.sum()
    log["unigram_val_nll"] = float(-p[val].log().mean())
json.dump(log, open(f"{OUT}/train_{name}.json", "w"), indent=1)
print("DONE", name, nparam, log["wall_s"])
