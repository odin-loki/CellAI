"""Wave 4 full queue: diagnostics → hyperband → E34 → E35 → E33."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
V7 = Path(__file__).resolve().parent / "run_arch_search_v7.py"


def run(cli: list[str], label: str) -> int:
    cmd = [sys.executable, str(V7), *cli]
    print("\n" + "=" * 72, label, " ".join(cmd), "=" * 72, flush=True)
    return subprocess.run(cmd, cwd=str(_REPO)).returncode


def main() -> int:
    steps = [
        (["--decode-sweep", "--ood-eval"], "Diagnostics"),
        (["--hyperband"], "Hyperband LR screen"),
        (["--train", "E34"], "E34 bypass retry"),
        (["--train", "E35"], "E35 doc-align"),
        (["--train", "E33"], "E33 E27+16k"),
    ]
    for cli, label in steps:
        if run(cli, label):
            return 1
    print("\nWave 4 queue finished.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
