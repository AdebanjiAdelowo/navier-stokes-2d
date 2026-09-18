"""Wall-clock runtime benchmark across grid resolutions.

Records hardware, precision, warm-up handling, and repetition count
explicitly, per the project's performance-reporting standard (measured
numbers only, no invented speedups).

Usage: python scripts/benchmark_runtime.py
"""
from __future__ import annotations

import pathlib
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import SimParams, SpectralGrid, run
from src.initial_conditions import random_vorticity_field

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"


def get_hardware_info() -> str:
    import subprocess

    cpu = "unknown"
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return (
        f"Platform: {platform.platform()}\n"
        f"CPU: {cpu}\n"
        f"Python: {platform.python_version()}\n"
        f"NumPy: {np.__version__}\n"
        f"Precision: float64 (complex128 in Fourier space), CPU only, single-threaded numpy.fft\n"
    )


def main() -> None:
    Ns = [32, 64, 128, 256]
    n_warmup_steps = 5
    n_timed_steps = 30
    n_repeats = 3

    lines = [get_hardware_info(), f"warm-up steps: {n_warmup_steps}, timed steps per repeat: "
             f"{n_timed_steps}, repeats: {n_repeats}\n",
             f"{'N':>5}  {'median ms/step':>16}  {'min ms/step':>13}  {'max ms/step':>13}"]

    for N in Ns:
        grid = SpectralGrid(N, N)
        omega0 = random_vorticity_field(grid, seed=0, k0=min(6.0, N / 8), target_enstrophy=1.0)
        p_warmup = SimParams(Nx=N, Ny=N, nu=0.01, dt=1e-3, t_end=n_warmup_steps * 1e-3, save_every=10**9)
        run(p_warmup, omega0)  # warm-up: population of any lazy caches, first-call overhead

        per_repeat_ms = []
        for _ in range(n_repeats):
            p = SimParams(Nx=N, Ny=N, nu=0.01, dt=1e-3, t_end=n_timed_steps * 1e-3, save_every=10**9)
            t0 = time.perf_counter()
            run(p, omega0)
            elapsed = time.perf_counter() - t0
            per_repeat_ms.append(1000.0 * elapsed / n_timed_steps)

        median_ms = float(np.median(per_repeat_ms))
        min_ms = float(np.min(per_repeat_ms))
        max_ms = float(np.max(per_repeat_ms))
        line = f"{N:5d}  {median_ms:16.3f}  {min_ms:13.3f}  {max_ms:13.3f}"
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "runtime_benchmark.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS_DIR / 'runtime_benchmark.txt'}")


if __name__ == "__main__":
    main()
