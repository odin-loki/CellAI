# Wave 2 follow-up: train E27, then E28, then E29; re-eval baselines.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
V5 = Path(__file__).resolve().parent / "run_arch_search_v5.py"


def run_step(cli: list[str], label: str) -> int:
    cmd = [sys.executable, str(V5), *cli]
    print("\n" + "=" * 72)
    print(label)
    print(" ", " ".join(cmd))
    print("=" * 72 + "\n", flush=True)
    return int(subprocess.run(cmd, cwd=str(_REPO)).returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Wave 2 jobs in sequence.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        choices=("all", "e27", "e28", "e29", "reeval"),
        default="all",
    )
    a = ap.parse_args()
    if not V5.exists():
        print("ERROR: missing", V5, file=sys.stderr)
        return 1

    steps: list[tuple[list[str], str]] = []
    if a.only in ("all", "e27"):
        steps.append((["--train", "E27"], "E27: D=1024 resume +8k"))
    if a.only in ("all", "e28"):
        steps.append((["--train", "E28"], "E28: D=512 resume +16k"))
    if a.only in ("all", "e29"):
        steps.append((["--train", "E29"], "E29: continuous EMA fork"))
    if a.only in ("all", "reeval"):
        steps.append(
            (["--reeval", "E25", "--no-gen"], "Re-eval E25 under protocol v1"),
        )

    if a.dry_run:
        print("DRY RUN:")
        for _, label in steps:
            print(" ", label)
        return 0

    for cli, label in steps:
        r = run_step(cli, label)
        if r:
            return r
    print("\nrun_wave2_followup: finished OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
