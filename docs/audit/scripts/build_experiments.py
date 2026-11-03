"""Build data/experiments.csv and figures 6-7 from the experiment ladder (values transcribed from
ARCH_SEARCH_PAPER.md, FAILURE_LOG and recovered raw results). Usage: python build_experiments.py OUT_DIR"""
import csv, math, sys
M=sys.argv[1]
# id, wave, round_label, change, D, steps, params_M, macro_nll, macro_ppl, text_code_nll, ms_call, outcome, source, notes
R=[
("E0","R1","Dense PDE + full Hebbian baseline (multi-domain)",256,"2k",25.8,19.3,2.38e8,None,1.845,"baseline","ASP §5.1,§6,§14","§5.1 reports macro NLL 13.415 (PPL 669,720); tables use 19.3 = final TRAIN loss. 966,000x = 238M/246.6 (238M = exp(19.29))"),
("E1","R1","SpectralPDE replaces dense PDE",256,"2k",25.8,12.295,2.19e5,None,1.976,"success","ASP §5.2,§6",""),
("E2","R1","SparseHebbian top-k/8",256,"2k",25.8,15.094,3.6e6,None,2.1,"partial","ASP §5.3,§14","later flagged as bug (hebb_rate too low/deferred)"),
("E3","R1","MultiScale partitions 4x128 + 2x256 K=8",256,"2k",26.2,11.742,1.26e5,None,2.3,"success","ASP §5.4","best R1"),
("E4","R1","AnnealedRouter (routing head)",256,"3k",26.8,None,None,None,1.845,"failure","ASP §5.5,§7.3","routing acc 0.320 (chance 0.333)"),
("E5","R1","PerFreqResonance (additive)",256,"2k",26.0,13.415,6.70e5,None,2.252,"failure","ASP §5.6","PPL 669,720 == exp(13.415) == E0's true macro: resonance never trained (Bug 1)"),
("E6","R1","Dense PDE scale D=512 N=8",512,"2k",52.7,None,1.07e13,None,2.081,"failure","ASP §5.7","diverged; PPL capped at exp(30)"),
("E7","R1","Spectral+Sparse+PerFreq D=512",512,"2k",52.4,41.317,1.07e13,None,2.599,"failure","ASP §5.8","PPL capped at exp(30)"),
("E8","R2","Spectral + MultiScale",256,"2k",26.2,9.313,11083,None,2.888,"success","ASP §10.2",""),
("E9","R2","Spectral + PerFreq (mult. gate)",256,"2k",26.0,12.295,2.19e5,None,2.839,"failure","ASP §10.3","bit-identical metrics to E1 -> resonance had no effect (Bug 1)"),
("E10","R2","Spectral + Sparse (deferred update)",256,"2k",25.9,28.073,1.56e12,None,2.444,"failure","ASP §10.4",""),
("E11","R2","Adaptive AnnealedRouter",256,"3k",None,None,None,None,None,"failure","ASP §10.5","routing = chance"),
("E12","R2","Spectral + Multi + PerFreq",256,"2k",26.3,9.313,11083,None,3.636,"failure","ASP §10.6","bit-identical metrics to E8 -> resonance had no effect (Bug 1)"),
("E13","R2","E12 + decoding fixes (rep penalty 1.3, noise 0.03)",256,"0",None,None,None,None,None,"success","ASP §10.7","diversity 0.993; decode-time only, no PPL"),
("E14","R2","Spectral + Multi D=512 4k",512,"4k",53.8,6.211,498,None,3.455,"success","ASP §10.8",""),
("E15","R3","Spectral+Multi+PerFreq (bugs fixed)",256,"2k",26.3,9.269,10603,None,3.056,"neutral","ASP §13.2","-0.044 vs E8"),
("E16","R3","Spectral+Multi+Sparse (hebb_rate 0.005)",256,"2k",26.2,6.760,863,None,2.861,"success","ASP §13.3","best D=256"),
("E17","R3","Domain-token routing",256,"2k",26.2,None,None,None,None,"failure","ASP §13.4","routing acc 0.335"),
("E18","R3","Spectral+Multi D=512 8k",512,"8k",53.5,5.774,322,None,2.705,"success","ASP §13.5",""),
("E19","R3","Spectral+Multi+Sparse+PerFreq",256,"2k",26.3,7.172,1303,None,3.312,"failure","ASP §13.6","worse than E16; raw results_v3 E19_routing acc 0.345"),
("E20","R3","Spectral+Multi+Sparse D=512 8k (champion R3)",512,"8k",53.5,5.508,246.6,7.964,3.180,"success","ASP §13.7,§17","§13.7/§14 say macro NLL 5.190 but ln(246.6)=5.508 (as §17 states)"),
("E21","R4","Continuous-context training (unshuffled)",256,"4k",26.2,7.229,1379,8.067,None,"partial","ASP §17.1","warm stream-matched eval; cold NLL 1940.9. Macro 7.229392 vs E25 7.229248 in raw results: near-coincidence, per-domain values differ"),
("E22","R4","Resume E20 +8k fine-tune",512,"16k",53.5,5.428,227.7,7.893,None,"success","ASP §17.2","re-evaluated 7.878 under Wave-2 split; raw results entry hand-entered (rounded values + note); eval set ~85% inside E20 training split (~45% visited)"),
("E23","R4","D=1024 4k",1024,"4k",108.0,5.605,272.5,8.053,None,"failure","ASP §17.3",""),
("E24","R4","D=1024 8k",1024,"8k",108.0,5.488,241.7,7.954,None,"neutral","ASP §17.4","beats E20, not E22"),
("E25","R4","Continuous D=512 shuffled docs",512,"4k",53.5,7.229,1379,8.043,None,"partial","ASP §17.5","warm eval; continuous-regime baseline"),
("E26","R4","Resume E25 +4k continuous",512,"8k",53.5,7.602,2002,8.549,None,"failure","ASP §17.6","warm PPL regressed 45%; paper reports text+code 8.049 but its per-domain values (8.718, 8.379) average 8.549"),
("E27","W2","Resume E24 +8k (D=1024, 16k total)",1024,"16k",108.0,None,220.1,7.847,None,"success","ASP §20.1-2","EVAL SPLIT CHANGED (seed 42): ~85% of eval docs in E24's training set"),
("E28","W2","Resume E22 +16k (D=512)",512,"32k",53.5,None,None,7.863,None,"neutral","ASP §20.2","contaminated split"),
("E29","W2","E25 fork + EMA (continuous)",512,"8k",53.5,None,None,7.948,None,"success","ASP §20.1-2","continuous champion; contaminated split"),
("E30","W3","Encoder->state bypass (resume E27)",1024,"+4k",109.0,None,None,133.6,None,"failure","ASP §20.3, FAILURE_LOG","committed code zero-inits bypass yet log blames random init: not reproducible from HEAD"),
("E31","W3","Input-side domain PDE banks",1024,"+4k",None,None,None,7.864,None,"neutral","ASP §20.3","routing 24.7% (< chance 33%)"),
("E32","W3","PerFreqResonance at D=1024",1024,"+4k",None,None,None,7.870,None,"neutral","ASP §20.3",""),
("E33","W4","E27 +16k early-stop",1024,"+3.2k",108.0,None,None,7.859,None,"failure","ASP §20.4","early-stopped on eval set (n=30)"),
("E34","W4","Zero-init bypass, freeze 1k + full 3k",1024,"+4k",109.0,None,None,7.300,None,"success","ASP §20.4","reset champion; bypass = learned zero-init projection of current-token embedding into output state (bigram path)"),
("E35","W4","Doc-align cosine aux loss",1024,"+4k",108.0,None,None,7.864,None,"neutral","ASP §20.4",""),
("H3","W4","Hyperband LR screen 600 steps @5e-5",1024,"+0.6k",108.0,None,None,7.646,None,"diagnostic","ASP §20.4","n=30 (10/modality), not comparable to n=150 headline"),
("OOD","W4","E27 'OOD' eval (reshuffle of same eval set, seed 99)",1024,"-",108.0,None,None,7.847,None,"diagnostic","ASP §20.4","not OOD: same eval pool; identical to E27 to 3 dp"),
]
cols=["id","wave","change","D","train_steps","params_M","macro_nll","macro_ppl","text_code_nll","ms_per_call","outcome","source","notes"]
with open(f"{M}/experiments.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(cols)
    for r in R: w.writerow([("" if v is None else v) for v in r])
import collections
c=collections.defaultdict(collections.Counter)
for r in R: c[r[1]][r[10]]+=1
print("wave outcome counts"); 
for k in ["R1","R2","R3","R4","W2","W3","W4"]:
    d=c[k]; n=sum(v for kk,v in d.items() if kk not in("baseline","diagnostic"))
    print(k,dict(d),"success rate %.0f%%"%(100*d["success"]/n if n else 0))
tot=collections.Counter(); [tot.update(v) for v in c.values()]; print("TOTAL",dict(tot))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ids=[r[0] for r in R]
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
x=[];y=[];best=[];b=1e9
for i,r in enumerate(R[:27]):
    if r[6] is not None and r[0] not in("E21","E25","E26"):
        x.append(i);y.append(r[6]);b=min(b,r[6]);best.append(b)
ax[0].scatter(x,y,c="#4C72B0",zorder=3,label="reported macro NLL (reset eval)")
ax[0].step(x,best,where="post",c="#DD8452",label="best so far")
ax[0].axhline(math.log(100277),ls="--",c="grey",lw=1); ax[0].text(0,math.log(100277)+0.6,"uniform guess ln(100277)=11.5",fontsize=8,color="grey")
ax[0].set_xticks(x); ax[0].set_xticklabels([ids[i] for i in x],rotation=60,fontsize=8)
ax[0].set_ylabel("macro NLL (nats/token)"); ax[0].set_title("Rounds 1-4: macro NLL (math-inflated)"); ax[0].legend(fontsize=8)
sub=[r for r in R if r[8] is not None and r[0] not in ("E30","OOD","E21","E25","E26","E29")]
xs=range(len(sub)); ys=[r[8] for r in sub]; bb=[];b=1e9
for v in ys: b=min(b,v); bb.append(b)
ax[1].scatter(xs,ys,c="#4C72B0",zorder=3,label="text+code NLL (reset)"); ax[1].step(xs,bb,where="post",c="#DD8452",label="best so far")
ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([r[0] for r in sub],rotation=60,fontsize=8)
ax[1].set_ylabel("text+code NLL"); ax[1].set_title("Honest headline: text+code NLL (E30=133.6 omitted)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{M}/best_so_far.png",dpi=150)
fig,ax=plt.subplots(figsize=(7,4))
waves=["R1","R2","R3","R4","W2","W3","W4"]; outs=["success","partial","neutral","failure","baseline","diagnostic"]
colors={"success":"#55A868","partial":"#CCB974","neutral":"#8C8C8C","failure":"#C44E52","baseline":"#4C72B0","diagnostic":"#8172B2"}
bottom=[0]*len(waves)
for o in outs:
    vals=[c[wv][o] for wv in waves]; ax.bar(waves,vals,bottom=bottom,color=colors[o],label=o); bottom=[a+b for a,b in zip(bottom,vals)]
ax.set_ylabel("experiments"); ax.set_title("Outcome by round/wave (author verdicts, normalised)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{M}/outcomes_by_wave.png",dpi=150)
