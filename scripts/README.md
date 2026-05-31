# Scripts

Runnable utilities from the **repository root** (paths resolve to `data/local/...` automatically).

| Script | Role |
|--------|------|
| `cli.py` | Source for the `cell-ai` console entry (`pip install -e .`) |
| `run_full_pipeline.py` | Data download + multi-domain training pipeline |
| `run_multimodal.py` | Multimodal router + NTP training |
| `run_eval.py` | Post-checkpoint evaluation and analysis |
| `_test_full.py` | Full benchmark / integration timing suite |
| `_profile_after.py`, `_profile_baseline.py` | Micro-benchmarks for core ops |
| `smoke_test_v2.py`, `smoke_test_v3.py` | Quick architecture sanity checks |

Architecture search experiments live in **`arch_search/`** (use `python -m arch_search.…`).
