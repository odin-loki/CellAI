from common import *
import numpy as np
from cellai_core.base import CellularPDE
from cellai_core.spectral_pde import SpectralPDE
from cellai_core.memory import MetaplasticityLayer
from cellai_core.sparse_hebbian import SparseHebbian
res = {"pde_vs_D": [], "hebb_vs_D": [], "hebb_vs_k": [], "seq_len": []}
Ds = [64,128,256,512,1024,2048,4096]
for D in Ds:
    p = ModelParams(device="cpu", num_partitions=4, state_size=D)
    for kind, mod in (("dense", CellularPDE(4, D)), ("spectral", SpectralPDE(4, D))):
        S = torch.randn(4, D); x = torch.randn(D)
        with torch.no_grad():
            f = timeit(lambda: mod.step(S, x, p), reps=200, warmup=50)["median_ms"]
        def fb():
            S2 = torch.randn(4, D, requires_grad=True); mod.zero_grad(); mod.step(S2, x, p).sum().backward()
        b = timeit(fb, reps=100, warmup=20)["median_ms"]
        np_ = sum(q.numel() for q in mod.parameters())
        live = sum(q.numel() for n,q in mod.named_parameters() if "input_gate" not in n)
        res["pde_vs_D"].append(dict(D=D, kind=kind, fwd_ms=f, fwdbwd_ms=b, params=np_, params_used_in_step=live))
        print("pde", D, kind, round(f,4), round(b,4), np_, live, flush=True)
    for kind in ("full", "sparse"):
        mod = MetaplasticityLayer(D) if kind == "full" else SparseHebbian(D, k_frac=0.125, defer_update=False)
        mod.train()
        a, b_, c = torch.randn(D), torch.randn(D), torch.randn(D)
        with torch.no_grad():
            f = timeit(lambda: mod(a, b_, c), reps=200, warmup=50)["median_ms"]
        res["hebb_vs_D"].append(dict(D=D, kind=kind, fwd_ms=f))
        print("hebb", D, kind, round(f,4), flush=True)
D = 1024
for kf in [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0]:
    mod = SparseHebbian(D, k_frac=kf, defer_update=False); mod.train()
    a, b_, c = torch.randn(D), torch.randn(D), torch.randn(D)
    with torch.no_grad():
        f = timeit(lambda: mod(a, b_, c), reps=200, warmup=50)["median_ms"]
    res["hebb_vs_k"].append(dict(D=D, k=mod.k, k_frac=kf, fwd_ms=f))
    print("k", mod.k, round(f,4), flush=True)
# Sequence length: inference forward over T tokens (cellular steps + tied LM head per token)
for name in ("v1_default", "v3_E20"):
    m = build(name); m.train()
    Dm = m.params.state_size
    for T in [64,128,256,512,1024,2048]:
        X = torch.randn(T, Dm)
        def run():
            m.partitions.reset()
            for t in range(T): m.cellular_step(X[t])
        with torch.no_grad():
            ms_ = timeit(run, reps=3, warmup=1)["median_ms"]
        res["seq_len"].append(dict(model=name, T=T, ms=ms_))
        print("seq", name, T, round(ms_,1), flush=True)
def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
fits = {}
for kind in ("dense","spectral"):
    r = [x for x in res["pde_vs_D"] if x["kind"]==kind]
    fits[f"pde_{kind}_fwd_slope_all"] = slope([x["D"] for x in r],[x["fwd_ms"] for x in r])
    rr = [x for x in r if x["D"]>=512]
    fits[f"pde_{kind}_fwd_slope_D>=512"] = slope([x["D"] for x in rr],[x["fwd_ms"] for x in rr])
    fits[f"pde_{kind}_fwdbwd_slope_D>=512"] = slope([x["D"] for x in rr],[x["fwdbwd_ms"] for x in rr])
for kind in ("full","sparse"):
    r = [x for x in res["hebb_vs_D"] if x["kind"]==kind and x["D"]>=512]
    fits[f"hebb_{kind}_slope_D>=512"] = slope([x["D"] for x in r],[x["fwd_ms"] for x in r])
fits["hebb_sparse_vs_k_slope"] = slope([x["k"] for x in res["hebb_vs_k"]],[x["fwd_ms"] for x in res["hebb_vs_k"]])
for name in ("v1_default","v3_E20"):
    r = [x for x in res["seq_len"] if x["model"]==name]
    fits[f"seq_{name}_slope"] = slope([x["T"] for x in r],[x["ms"] for x in r])
res["fits"] = fits
print(json.dumps(fits, indent=1))
json.dump(res, open(f"{OUT}/sweeps.json","w"), indent=1)
