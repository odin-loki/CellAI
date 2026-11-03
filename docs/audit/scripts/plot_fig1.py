"""Figure 1: CellAI text+code NLL vs count-based baselines, per evaluation split.
Usage: python plot_fig1.py out.png
Split A = drivers v1-v4 (E21-E26); split B = v3_resume and v5-v7 (E20, E27-E35).
Baselines from data/ngram_split_A_v1-v4.json and data/ngram_split_B_v5-v7.json (50 docs/modality).
CellAI values as reported (ARCH_SEARCH_PAPER.md section 17-20, raw results_v4.json)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_base = [("Trigram (KN)", 4.991), ("Bigram (KN)", 5.522), ("Unigram", 7.907)]
B_base = [("Trigram (KN)", 4.895), ("Bigram (KN)", 5.458), ("Unigram", 7.868)]
# (label, value, contaminated, warm-protocol)
A_runs = [("E22  E20+8k", 7.893, True, False), ("E24  D=1024 8k", 7.954, False, False),
          ("E23  D=1024 4k", 8.053, False, False), ("E25  continuous", 8.043, False, True),
          ("E21  continuous", 8.067, False, True), ("E26  continuous", 8.549, False, True)]
B_runs = [("E34  bypass", 7.300, True, False), ("E27  D=1024 16k", 7.847, True, False),
          ("E33", 7.859, True, False), ("E28", 7.863, True, False), ("E31", 7.864, True, False),
          ("E35", 7.864, True, False), ("E32", 7.870, True, False),
          ("E29  continuous+EMA", 7.948, True, True), ("E20  'champion'", 7.964, False, False)]
CELL, UNI, BASE = "#2a78d6", "#eb6834", "#8a8984"

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=200)
for ax, (title, base, runs) in zip(axes, [("Split A (drivers v1–v4)", A_base, A_runs),
                                         ("Split B (v3-resume, v5–v7)", B_base, B_runs)]):
    rows = [(n, v, "uni" if n == "Unigram" else "base", False, False) for n, v in base[:2]]
    rows += [(n, v, "cell", c, w) for n, v, c, w in sorted(runs, key=lambda r: r[1])]
    uni = base[2][1]
    rows.insert(2, ("Unigram", uni, "uni", False, False))
    rows = sorted(rows, key=lambda r: r[1])
    y = list(range(len(rows)))[::-1]
    for yi, (n, v, k, c, w) in zip(y, rows):
        color = {"cell": CELL, "uni": UNI, "base": BASE}[k]
        ax.barh(yi, v, color=color, height=0.62, hatch="///" if c else None,
                edgecolor="white" if c else color, linewidth=0)
        ax.text(v + 0.06, yi, f"{v:.2f}", va="center", fontsize=7.5)
    labels = [n + (" *" if c else "") + (" (warm)" if w else "") for n, v, k, c, w in rows]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(uni, color=UNI, lw=1, ls="--")
    ax.set_xlim(0, 9.4); ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel("Held-out text+code NLL (nats/token, lower is better)", fontsize=8)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
fig.suptitle("CellAI runs vs count-based baselines on the same split, data and protocol", fontsize=10.5, x=0.01, ha="left")
fig.text(0.01, 0.012,
         "* hatched: eval set ~85% inside an ancestor checkpoint's training split (~45% actually trained on). "
         "(warm): stream-matched continuous protocol. 50-doc sampling s.d.: ~0.05 (unigram), ~0.1 (bigram).",
         fontsize=6.5, color="#52514e")
plt.tight_layout(rect=(0, 0.035, 1, 0.95))
plt.savefig(sys.argv[1] if len(sys.argv) > 1 else "fig1_baselines.png")
