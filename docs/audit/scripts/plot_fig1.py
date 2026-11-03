import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
rows=[("Trigram (Kneser-Ney)",4.895,"base"),("Bigram (Kneser-Ney)",5.458,"base"),
("E34 bypass (reset champion)",7.300,"cell"),("E27 D=1024, 16k",7.847,"cell"),
("Unigram (token frequencies only)",7.868,"uni"),
("E22 E20+8k *",7.893,"cell"),("E29 continuous+EMA (warm)",7.948,"cell"),("E24 D=1024, 8k",7.954,"cell"),
("E20 'champion' D=512",7.964,"cell"),("E25 continuous (warm)",8.043,"cell")]
col={"cell":"#2a78d6","base":"#8a8984","uni":"#eb6834"}
fig,ax=plt.subplots(figsize=(8,5),dpi=200)
y=list(range(len(rows)))[::-1]
for yi,(n,v,k) in zip(y,rows):
    ax.barh(yi,v,color=col[k],height=0.62)
    ax.text(v+0.06,yi,f"{v:.2f}",va="center",fontsize=8.5,color="#0b0b0b")
ax.axvline(7.868,color="#eb6834",lw=1,ls="--")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=8.5)
ax.set_xlim(0,9); ax.set_xlabel("Held-out text+code NLL (nats/token, lower is better)",fontsize=9)
fig.suptitle("CellAI champions vs. count-based baselines, same data & protocol",fontsize=10.5,x=0.02,ha="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x",color="#e6e5e0",lw=0.6); ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=col["cell"],label="CellAI (reported)"),Patch(color=col["uni"],label="Unigram (this audit)"),Patch(color=col["base"],label="n-gram (this audit)")],
          fontsize=7.5,loc="lower left",bbox_to_anchor=(0,1.0),ncol=3,frameon=False)
fig.text(0.01,0.01,"* E22 is evaluated on documents ~84% of which were in its parent checkpoint's training set. 50-doc sampling s.d.: ~0.05 (unigram), ~0.11 (bigram).",fontsize=6.5,color="#52514e")
plt.tight_layout(rect=(0,0.03,1,0.95)); plt.savefig("figs/fig_baselines.png")
