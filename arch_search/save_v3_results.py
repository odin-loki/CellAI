"""Save E15/E16 results from checkpoints into results_v3.json."""
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT_DIR = _REPO / "data" / "local" / "arch_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)
results = {}
if (OUT_DIR / "results_v3.json").exists():
    with open(OUT_DIR / "results_v3.json") as f:
        results = json.load(f)
    print(f"Existing keys: {list(results.keys())}")
else:
    print("No existing v3 results - creating fresh")

for exp, fn in [
    ("E15", "E15_SpectralMultiResonanceFixed.pt"),
    ("E16", "E16_SpectralMultiSparseFixed.pt"),
]:
    p = OUT_DIR / fn
    if p.exists():
        ck = torch.load(str(p), map_location="cpu")
        r = ck.get("result", {})
        results[exp] = r
        tr = r.get("train", {})
        ppl = r.get("ppl", {})
        print(
            f"Loaded {exp}: final={tr.get('final', '?')}  macro_nll={ppl.get('macro_avg_nll', '?')}  macro_ppl={ppl.get('macro_ppl', '?')}"
        )
    else:
        print(f"  Checkpoint not found: {p}")

with open(OUT_DIR / "results_v3.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
rel = OUT_DIR.relative_to(_REPO)
print(f"Saved -> {rel / 'results_v3.json'}")
