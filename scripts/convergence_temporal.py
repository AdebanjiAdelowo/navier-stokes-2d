"""Temporal convergence study using the two-mode manufactured solution.

Resolution is fixed high enough that spatial error is negligible (the
manufactured wavenumbers are fully resolved), so the measured error is
purely from the RK4 time-stepping of the nonlinear term (the linear/viscous
part is integrated exactly by the integrating factor at any dt). Produces
figures/convergence_temporal.png and results/convergence_temporal.txt.

Usage: python scripts/convergence_temporal.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import SimParams, SpectralGrid, run, ManufacturedSolution

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"


def run_case(N: int, dt: float, t_end: float, ms: ManufacturedSolution) -> float:
    grid = SpectralGrid(N, N)
    omega0 = ms.omega_exact(grid.X, grid.Y, 0.0)
    p = SimParams(Nx=N, Ny=N, nu=ms.nu, dt=dt, t_end=t_end, save_every=10**9, dealias=True)

    def forcing(t):
        return grid.fft2(ms.forcing(grid.X, grid.Y, t))

    res = run(p, omega0, forcing=forcing)
    exact_final = ms.omega_exact(grid.X, grid.Y, res.t[-1])
    return float(np.linalg.norm(res.omega[-1] - exact_final) / np.linalg.norm(exact_final))


def main() -> None:
    ms = ManufacturedSolution()  # k1=(1,1), k2=(2,1): fully resolved at N=64
    N = 64
    t_end = 0.2
    dts = [0.02, 0.01, 0.005, 0.0025, 0.00125]

    errs = [run_case(N, dt, t_end, ms) for dt in dts]

    print(f"{'dt':>10}  {'rel L2 error':>14}  {'observed order':>15}")
    lines = [f"{'dt':>10}  {'rel L2 error':>14}  {'observed order':>15}"]
    for i, (dt, err) in enumerate(zip(dts, errs)):
        if i == 0:
            order_str = "--"
        else:
            order = np.log(errs[i - 1] / errs[i]) / np.log(dts[i - 1] / dts[i])
            order_str = f"{order:.3f}"
        line = f"{dt:10.5f}  {err:14.4e}  {order_str:>15}"
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "convergence_temporal.txt").write_text(
        "Temporal convergence, two-mode manufactured solution, N=64 (fully resolved), "
        f"t_end={t_end}\n" + "\n".join(lines) + "\n"
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(dts, errs, "o-", color="#1f77b4", label="observed error")
    # reference 4th-order slope anchored at the coarsest, best-resolved point
    ref = errs[0] * (np.array(dts) / dts[0]) ** 4
    ax.loglog(dts, ref, "k--", label=r"$O(\Delta t^4)$ reference")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Temporal convergence (manufactured solution, N=64)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "convergence_temporal.png", dpi=150)
    print(f"\nWrote {FIG_DIR / 'convergence_temporal.png'} and {RESULTS_DIR / 'convergence_temporal.txt'}")


if __name__ == "__main__":
    main()
