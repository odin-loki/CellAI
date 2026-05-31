"""
data.config
===========
Central data configuration.

Set the DATA_ROOT environment variable to point at your 6TB drive.
All pipelines read this value to know where to download/write data.

Usage:
    export DATA_ROOT=E:\\cellai_data   # Windows
    export DATA_ROOT=/mnt/cellai_data # Linux

Or create a .env file (copy .env.example) and it will be loaded automatically.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell env

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/local"))

NLP_RAW        = DATA_ROOT / "nlp" / "raw"
NLP_PROCESSED  = DATA_ROOT / "nlp" / "processed"
NLP_TRAIN      = NLP_PROCESSED / "train.jsonl"
NLP_VAL        = NLP_PROCESSED / "val.jsonl"
NLP_TEST       = NLP_PROCESSED / "test.jsonl"

CODE_RAW       = DATA_ROOT / "code" / "raw"
CODE_PROCESSED = DATA_ROOT / "code" / "processed"

MATH_DIR       = DATA_ROOT / "math" / "generated"
MATH_TRAIN     = MATH_DIR / "train.jsonl"
MATH_VAL       = MATH_DIR / "val.jsonl"

CHECKPOINTS    = DATA_ROOT / "checkpoints"


def ensure_dirs() -> None:
    """Create all data directories if they do not exist."""
    for d in [NLP_RAW, NLP_PROCESSED, CODE_RAW, CODE_PROCESSED, MATH_DIR, CHECKPOINTS]:
        d.mkdir(parents=True, exist_ok=True)


def print_config() -> None:
    print(f"DATA_ROOT   : {DATA_ROOT}")
    print(f"  nlp/raw   : {NLP_RAW}")
    print(f"  nlp/proc  : {NLP_PROCESSED}")
    print(f"  code/raw  : {CODE_RAW}")
    print(f"  code/proc : {CODE_PROCESSED}")
    print(f"  math      : {MATH_DIR}")
    print(f"  ckpts     : {CHECKPOINTS}")
    free_gb = _free_gb(DATA_ROOT)
    if free_gb is not None:
        print(f"  free GB   : {free_gb:.1f}")


def _free_gb(path: Path) -> float | None:
    try:
        import shutil
        total, used, free = shutil.disk_usage(path.parent if not path.exists() else path)
        return free / 1e9
    except Exception:
        return None
