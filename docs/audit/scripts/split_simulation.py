"""Reproduce the train/eval permutation each arch_search script builds.
Usage: python split_simulation.py run_arch_search_v4 > perm_v4.json
Each script is imported in its natural import order (import-time seeding matters)."""
import sys, json, os, importlib, io, contextlib, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
tmp = os.path.dirname(os.path.abspath(__file__)) + "/data"
fields = [("nlp","text",0),("code","content",1),("math","problem",2)]
# Dummy corpora: 6000 uniquely-identified records per domain, so overlap between splits can be measured.
os.makedirs(tmp, exist_ok=True)
for d, f, l in fields:
    p = f"{tmp}/{d}.jsonl"
    if not os.path.exists(p):
        with open(p, "w") as fh:
            for i in range(6000):
                fh.write(json.dumps({f: f"{d}-{i:05d} " + "x" * 40}) + "\n")
mode = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    mod = importlib.import_module("arch_search."+mode)   # natural import order
    import arch_search.run_arch_search as ras
    from pathlib import Path
    state = random.getstate()
    ras.MODALITY_FIELDS = [(Path(f"{tmp}/{d}.jsonl"), f, l) for d,f,l in fields]
    random.setstate(state)
    ds = ras.build_dataset(6000)
    out = {"perm1": [t[:10] for t,_ in ds]}
    if mode == "run_arch_search_v3":
        out["perm2_domain"] = [t[:10] for t,_ in ras.build_dataset(6000)]
print(json.dumps(out))
