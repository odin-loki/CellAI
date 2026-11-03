import json, glob, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "" + os.environ.get('AUDIT_OUT', 'audit_out') + ""
plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"dense": "#2a6fdb", "spectral": "#e07a1f", "full": "#2a6fdb", "sparse": "#e07a1f"}
sw = json.load(open(f"{OUT}/sweeps.json"))
# 1. PDE scaling
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for kind in ("dense", "spectral"):
    r = [x for x in sw["pde_vs_D"] if x["kind"] == kind]
    ax[0].loglog([x["D"] for x in r], [x["fwd_ms"] for x in r], "o-", color=C[kind], label=f"{kind} forward")
    ax[0].loglog([x["D"] for x in r], [x["fwdbwd_ms"] for x in r], "s--", color=C[kind], alpha=0.7, label=f"{kind} fwd+bwd")
    ax[1].loglog([x["D"] for x in r], [x["params"] for x in r], "o-", color=C[kind], label=f"{kind} params (declared)")
    if kind == "spectral":
        ax[1].loglog([x["D"] for x in r], [x["params_used_in_step"] for x in r], "^:", color=C[kind], label="spectral params actually used")
ax[0].set(xlabel="state size D (N=4 partitions)", ylabel="ms per step (CPU, 4 threads)", title="PDE step latency vs D")
ax[1].set(xlabel="state size D", ylabel="parameters", title="PDE parameter count vs D")
for a in ax: a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pde_scaling.png"); plt.close(fig)
# 2. Hebbian scaling
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for kind in ("full", "sparse"):
    r = [x for x in sw["hebb_vs_D"] if x["kind"] == kind]
    ax[0].loglog([x["D"] for x in r], [x["fwd_ms"] for x in r], "o-", color=C[kind], label=f"{'MetaplasticityLayer (full)' if kind=='full' else 'SparseHebbian k=D/8'}")
ax[0].set(xlabel="state size D", ylabel="ms per call", title="Hebbian layer latency vs D"); ax[0].legend(fontsize=8)
r = sw["hebb_vs_k"]
ax[1].semilogx([x["k"] for x in r], [x["fwd_ms"] for x in r], "o-", color=C["sparse"], label="measured")
ks = [x["k"] for x in r]; base = r[-1]["fwd_ms"]
ax[1].semilogx(ks, [base * (k/ks[-1])**2 for k in ks], ":", color="grey", label="if cost ∝ k² (claimed)")
ax[1].set(xlabel="top-k active units (D=1024)", ylabel="ms per call", title="SparseHebbian latency vs k", ylim=(0, 1.2*max(x["fwd_ms"] for x in r)))
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_hebbian_scaling.png"); plt.close(fig)
# 3. sequence length
fig, ax = plt.subplots(figsize=(5.5, 4))
for name, col in (("v1_default", "#2a6fdb"), ("v3_E20", "#e07a1f")):
    r = [x for x in sw["seq_len"] if x["model"] == name]
    ax.loglog([x["T"] for x in r], [x["ms"] for x in r], "o-", color=col, label=name)
ax.set(xlabel="sequence length T (tokens)", ylabel="ms (cellular steps only, no LM head)", title="Inference time vs sequence length")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_seq_scaling.png"); plt.close(fig)
# 4. param breakdown
rows = [json.loads(l) for l in open(f"{OUT}/bench_models.jsonl")]
fig, ax = plt.subplots(figsize=(8, 4))
names = [r["name"] for r in rows]
emb = [r["embedding_params"]/1e6 for r in rows]; rest = [r["non_embedding_params"]/1e6 for r in rows]
ax.barh(names, emb, color="#9bb8e8", label="token embedding (tied LM head)")
ax.barh(names, rest, left=emb, color="#e07a1f", label="everything else (PDE, Hebbian, memory, ...)")
for i, r in enumerate(rows):
    ax.text(r["total_params"]/1e6 + 1, i, f"{r['total_params']/1e6:.1f}M  ({100*(1-r['embedding_share']):.1f}% non-emb)", va="center", fontsize=8)
ax.set(xlabel="parameters (millions)", title="Where the parameters live", xlim=(0, max(r["total_params"] for r in rows)/1e6*1.45))
ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_param_breakdown.png"); plt.close(fig)
# 5. training curves
logs = {os.path.basename(f)[6:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/train_*.json"))}
if logs:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = ["#8c564b", "#555555", "#2a6fdb", "#9467bd", "#e07a1f", "#2ca02c", "#d62728"]
    for (n, L), c in zip(logs.items(), cols):
        ax[0].plot([x["step"]*64 for x in L["train"]], [x["nll"] for x in L["train"]], color=c, label=n, lw=1.2)
        ax[1].plot([x["step"]*64 for x in L["val"]], [x["nll"] for x in L["val"]], "o-", color=c, label=n, ms=3)
    uni = logs.get("transformer", {}).get("unigram_val_nll")
    if uni: ax[1].axhline(uni, ls=":", color="k", label=f"unigram baseline ({uni:.2f})")
    for a, t in zip(ax, ("Train NLL (25-step avg)", "Held-out NLL (24×64-token windows, reset state)")):
        a.set(xlabel="training tokens", ylabel="nats/token", title=t, ylim=(6.5, 16)); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_training_curves.png"); plt.close(fig)
print("ok")
