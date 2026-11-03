"""Figure 8: 24-h response to Kukushkin et al. (2024) protocols, normalised to the massed pulse."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
res = json.load(open(sys.argv[1])); out = sys.argv[2]
order = [("1x", "1×3 min"), ("2x_first_last", "2× (first+last)"), ("massed_12min", "massed 12 min"),
         ("4x_iti10", "4× ITI 10"), ("4x_iti20", "4× ITI 20"), ("4x_iti30", "4× ITI 30")]
panels = [("A_core_equation", "A. CellAI core equation\n(core_math_model §2.1.1)"),
          ("B_dense_pde_basal_0.1", "B. Implemented dense PDE\n(basal state 0.1)"),
          ("C_spectral_pde", "C. Implemented SpectralPDE\n(v3)"),
          ("D_reference_cascade", "D. Reference cascade\n(adapting kinase→ERK→CREB→luc)")]
col = {"massed_12min": "#eb6834"}
fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), dpi=200, sharey=False)
for ax, (key, title) in zip(axes, panels):
    r = res[key]; m = r["massed_12min"]["24h"]
    vals = [r[p]["24h"] / m for p, _ in order]
    ax.bar(range(len(order)), vals, color=[col.get(p, "#2a78d6") for p, _ in order], width=0.66)
    ax.axhline(1.0, color="#8a8984", lw=0.8, ls="--")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], rotation=55, ha="right", fontsize=7)
    ax.set_title(title, fontsize=8.5, loc="left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    if key == "D_reference_cascade":
        ax.set_yscale("log"); ax.set_ylim(0.1, 500)
    for i, v in enumerate(vals):
        ax.text(i, v * (1.15 if key == "D_reference_cascade" else 1.0) + (0 if key == "D_reference_cascade" else 0.02 * max(vals)),
                f"{v:.2g}", ha="center", va="bottom", fontsize=6.5)
axes[0].set_ylabel("24-h response ÷ massed response", fontsize=8)
fig.suptitle("Does CellAI reproduce the massed-spaced effect of its founding paper (Kukushkin et al. 2024)?",
             fontsize=10, x=0.01, ha="left")
fig.text(0.01, 0.005, "Bars > 1 at 4× = spacing effect (seen in cells: 4× ITI 10 > ITI 20 > ITI 30 > massed). "
         "Dense PDE from the zero state (not shown) gives only noise-level (~1e-7) responses. Panel D is illustrative, not fitted.",
         fontsize=6.5, color="#52514e")
plt.tight_layout(rect=(0, 0.04, 1, 0.92)); plt.savefig(out)
