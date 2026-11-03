"""In-silico massed-spaced test of CellAI's dynamics (Kukushkin et al., 2024 protocols).

Protocols (Kukushkin et al. 2024, Nat Commun 15:9635): a single 3-min pulse (1x); four 3-min
pulses with inter-trial interval (ITI) 10, 20 or 30 min (4x); a single 12-min "massed" pulse
(same total exposure as 4x); two 3-min pulses matching the first and last pulse of the 4x/ITI-10
protocol (2x). Readouts at 4 h and 24 h after onset. Here ITI is taken as onset-to-onset spacing.

Models:
  A. CellAI core equation, docs/math_models/core_math_model.md 2.1.1, with g(S) = S (scalar):
        dS/dt = sigma(I - theta) * S - gamma * S + eta(t)
     Analytically S(T) = S0 * exp(int (sigma(I - theta) - gamma) dt): depends only on total exposure,
     so it cannot distinguish spaced from massed training.
  B. Implemented dense CellularPDE (cellai_core/base.py), random init, N=4, D=64.
  C. Implemented SpectralPDE (cellai_core/spectral_pde.py), random init, N=4, D=64.
  D. Reference: a minimal adapting kinase -> ERK -> CREB (positive feedback) -> short-lived luciferase
     cascade. Parameters chosen by a small grid search for the qualitative ordering
     4x(ITI10) > 4x(ITI20) > 4x(ITI30) > massed at 24 h; not fitted to data.

Usage: python spacing_effect.py [out.json]
"""
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from cellai_core.base import CellularPDE, ModelParams  # noqa: E402
from cellai_core.spectral_pde import SpectralPDE  # noqa: E402

DT = 0.25                       # minutes per simulation step
T_END = 24 * 60                 # minutes
STEPS = int(T_END / DT)
READ = {"4h": int(240 / DT), "24h": STEPS - 1}


def protocol(name):
    s = np.zeros(STEPS)
    def pulse(t0, dur):
        s[int(t0 / DT):int((t0 + dur) / DT)] = 1.0
    if name == "control":
        pass
    elif name == "1x":
        pulse(0, 3)
    elif name.startswith("4x_iti"):
        iti = int(name.split("iti")[1])
        for k in range(4):
            pulse(k * iti, 3)
    elif name == "massed_12min":
        pulse(0, 12)
    elif name == "2x_first_last":
        pulse(0, 3); pulse(30, 3)
    return s


PROTOCOLS = ["1x", "4x_iti10", "4x_iti20", "4x_iti30", "massed_12min", "2x_first_last"]


def model_core_equation(s, S0=0.1, theta=2.0, gain=4.0):
    """Model A. Scalar CellAI core ODE with g(S)=S; Euler; per-minute rates.
    gamma is set to the resting value of sigma(-theta) so the unstimulated state is stationary."""
    gamma = 1 / (1 + math.exp(theta))
    S = S0; out = np.empty(STEPS)
    for t in range(STEPS):
        S += DT * ((1 / (1 + math.exp(-(gain * s[t] - theta)))) * S - gamma * S)
        out[t] = S
    return out


def model_torch_pde(kind, s, seed=0, D=64, N=4, amp=1.0, basal=0.0):
    """Models B/C. The repository's own PDE modules driven by a stimulus vector."""
    torch.manual_seed(seed)
    pde = CellularPDE(N, D) if kind == "dense" else SpectralPDE(N, D)
    params = ModelParams()
    u = torch.randn(D, generator=torch.Generator().manual_seed(seed + 1))
    u = u / u.norm() * math.sqrt(D) * amp           # stimulus direction, embedding-scale
    # Common random numbers: identical noise stream for treated and control runs.
    torch.manual_seed(seed + 2)
    S = torch.full((N, D), basal)
    out = np.empty(STEPS)
    with torch.no_grad():
        for t in range(STEPS):
            S = pde.step(S, u * float(s[t]), params)
            out[t] = S.mean(0).norm().item()
    return out


def model_cascade(s, tau_R=3.0, tau_E=12.0, kE=0.8, theta_E=0.75, kC=0.02, kfb=0.004, c0=0.5,
                  tau_C=360.0, tau_L=60.0):
    """Model D. Adapting kinase (incoherent feed-forward) -> ERK accumulation -> CREB with
    cooperative positive feedback -> PEST-luciferase. Minutes. Qualitative only."""
    R = E = C = L = 0.0
    out = np.empty(STEPS)
    for t in range(STEPS):
        K = max(s[t] - R, 0.0)                       # kinase drive adapts to sustained input
        R += DT * (s[t] - R) / tau_R
        E += DT * (kE * K * (1 - E) - E / tau_E)
        drive = E ** 4 / (E ** 4 + theta_E ** 4)
        C += DT * (kC * drive + kfb * C ** 2 / (C ** 2 + c0 ** 2) - C / tau_C)
        L += DT * (C - L) / tau_L
        out[t] = L
    return out


def run():
    res = {}
    models = {
        "A_core_equation": lambda s: model_core_equation(s),
        "B_dense_pde_zero_state": lambda s: model_torch_pde("dense", s),
        "B_dense_pde_basal_0.1": lambda s: model_torch_pde("dense", s, basal=0.1),
        "C_spectral_pde": lambda s: model_torch_pde("spectral", s),
        "D_reference_cascade": lambda s: model_cascade(s),
    }
    for mname, fn in models.items():
        ctrl = fn(protocol("control"))
        res[mname] = {}
        for p in PROTOCOLS:
            y = fn(protocol(p))
            res[mname][p] = {k: float(y[i] - ctrl[i]) for k, i in READ.items()}
        r = res[mname]
        m24 = r["massed_12min"]["24h"]
        r["_spaced_over_massed_24h"] = (r["4x_iti10"]["24h"] / m24) if abs(m24) > 1e-12 else None
    return res


if __name__ == "__main__":
    out = run()
    path = sys.argv[1] if len(sys.argv) > 1 else "spacing_effect_results.json"
    json.dump(out, open(path, "w"), indent=1)
    for m, r in out.items():
        print(m)
        for p in PROTOCOLS:
            print(f"   {p:15s} 4h={r[p]['4h']:+.4g}   24h={r[p]['24h']:+.4g}")
        print("   spaced(ITI10)/massed at 24h:", r["_spaced_over_massed_24h"])
