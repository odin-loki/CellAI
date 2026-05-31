"""cellai_core.utils — seeds, basic timing, CLI base."""
from __future__ import annotations
import random, time, cProfile, pstats, io
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_benchmark(fn, reps: int = 100, warmup: int = 10) -> dict:
    """Time `fn()` with GPU sync; return dict with mean ms and throughput."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / reps * 1000
    return {"ms_per_call": elapsed, "calls_per_s": 1000 / elapsed}


class CProfileContext:
    """Context manager wrapping cProfile for quick hotspot checks."""
    def __init__(self, sort: str = "cumulative", lines: int = 20):
        self.sort = sort
        self.lines = lines
        self._pr = cProfile.Profile()

    def __enter__(self):
        self._pr.enable()
        return self

    def __exit__(self, *_):
        self._pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(self._pr, stream=s).sort_stats(self.sort)
        ps.print_stats(self.lines)
        print(s.getvalue())
