# Round 4 follow-up: --ablation-burn, --train E21, --train E26 in order.
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
V4 = _SCRIPT_DIR / "run_arch_search_v4.py"
E25_CKPT = _REPO / "data" / "local" / "arch_search" / "E25_ContinuousD512.pt"


def run_step(cli: list[str], label: str) -> int:
    cmd = [sys.executable, str(V4), *cli]
    print("\n" + "=" * 72)
    print(label)
    print(" ", " ".join(cmd))
    print("=" * 72 + "\n", flush=True)
    return int(subprocess.run(cmd, cwd=str(_REPO)).returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Round 4 follow-up jobs in sequence.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=("all", "ablation", "e21", "e26"), default="all")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-e21", action="store_true")
    ap.add_argument("--skip-e26", action="store_true")
    a = ap.parse_args()
    if not V4.exists():
        print("ERROR: missing", V4, file=sys.stderr)
        return 1
    do_ab = not a.skip_ablation and a.only in ("all", "ablation")
    do_e21 = not a.skip_e21 and a.only in ("all", "e21")
    do_e26 = not a.skip_e26 and a.only in ("all", "e26")
    if a.dry_run:
        print("DRY RUN - planned:")
        if do_ab:
            print("  [1] --ablation-burn")
        if do_e21:
            print("  [2] --train E21")
        if do_e26:
            print("  [3] --train E26")
        return 0
    if do_ab:
        if not E25_CKPT.exists():
            print("[SKIP ablation-burn] missing", E25_CKPT)
        else:
            r = run_step(["--ablation-burn"], "Step 1/3: ablation-burn")
            if r:
                return r
    if do_e21:
        r = run_step(["--train", "E21"], "Step 2/3: train E21")
        if r:
            return r
    if do_e26:
        if not E25_CKPT.exists():
            print("[SKIP E26] missing", E25_CKPT)
        else:
            r = run_step(["--train", "E26"], "Step 3/3: train E26")
            if r:
                return r
    print("\nrun_round4_followup: finished OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
