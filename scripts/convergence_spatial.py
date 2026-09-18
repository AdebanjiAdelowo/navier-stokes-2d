"""Spatial convergence / resolution study.

Two parts, because a spectral method's spatial error behaves differently
from a finite-difference method's and a single figure cannot honestly show
both regimes:

(A) Manufactured solution, band-limited (only wavenumbers up to k=6 present).
    A pseudo-spectral method has NO fixed polynomial spatial order for
    smooth periodic data: once N is large enough that the 2/3-dealiasing
    cutoff exceeds every wavenumber present in the data, the solution is
    represented exactly and the error collapses to machine precision.
    Below that resolution threshold the field is under-resolved/aliased,
    which does not produce a smooth power-law error curve (aliasing folds
    energy unpredictably depending on N mod k). This is expected, correct
    spectral-method behaviour, not a bug, and is reported honestly below
    rather than dressed up as an algebraic convergence rate.

(B) Broadband decaying-turbulence initial condition (energy spread over a
    continuum of scales, no exact solution). Self-convergence against a
    high-resolution reference solution shows the smoother, more familiar
    "error decreases as resolution increases" curve, which is the
    appropriate way to assess resolution requirements for a field that is
    not exactly band-limited.

Produces figures/convergence_spatial_mms.png, figures/convergence_spatial_turbulence.png,
and results/convergence_spatial.txt.

Usage: python scripts/convergence_spatial.py
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
from src.initial_conditions import random_vorticity_field

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"


def mms_resolution_sweep() -> list[str]:
    ms = ManufacturedSolution(k1x=3, k1y=4, k2x=6, k2y=2, nu=0.05)
    t_end = 0.02
    dt = 5e-5  # tiny, to make temporal error negligible relative to spatial error
    Ns = [8, 12, 16, 20, 24, 32, 48]

    errs = []
    for N in Ns:
        grid = SpectralGrid(N, N)
        omega0 = ms.omega_exact(grid.X, grid.Y, 0.0)
        p = SimParams(Nx=N, Ny=N, nu=ms.nu, dt=dt, t_end=t_end, save_every=10**9, dealias=True)

        def forcing(t, grid=grid):
            return grid.fft2(ms.forcing(grid.X, grid.Y, t))

        res = run(p, omega0, forcing=forcing)
        exact_final = ms.omega_exact(grid.X, grid.Y, res.t[-1])
        err = float(np.linalg.norm(res.omega[-1] - exact_final) / np.linalg.norm(exact_final))
        errs.append(err)

    # resolvable wavenumber under the 2/3 rule is (2/3)*(N/2); the highest
    # manufactured wavenumber is 6, so full resolution needs N > 18
    lines = [
        "Part A: manufactured solution (k up to 6), dt=5e-5 (temporal error negligible)",
        f"{'N':>4}  {'2/3-rule cutoff':>16}  {'rel L2 error':>14}",
    ]
    for N, err in zip(Ns, errs):
        cutoff = (2.0 / 3.0) * (N / 2.0)
        lines.append(f"{N:4d}  {cutoff:16.2f}  {err:14.4e}")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.semilogy(Ns, errs, "o-", color="#d62728")
    ax.axvline(18, color="k", linestyle="--", alpha=0.6, label="full-resolution threshold (N=18)")
    ax.set_xlabel("N (grid points per direction)")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Spatial resolution: band-limited manufactured solution")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_spatial_mms.png", dpi=150)

    return lines


def turbulence_self_convergence() -> list[str]:
    k0 = 6.0
    nu = 0.02
    t_end = 0.1
    dt_ref = 2e-4
    N_ref = 256
    Ns = [16, 24, 32, 48, 64, 96]

    grid_ref = SpectralGrid(N_ref, N_ref)
    omega0_ref = random_vorticity_field(grid_ref, seed=7, k0=k0, target_enstrophy=1.0)
    p_ref = SimParams(Nx=N_ref, Ny=N_ref, nu=nu, dt=dt_ref, t_end=t_end, save_every=10**9)
    res_ref = run(p_ref, omega0_ref)
    omega_ref_hat = grid_ref.fft2(res_ref.omega[-1])

    lines = [
        "",
        f"Part B: broadband decaying-turbulence self-convergence, reference N={N_ref}, "
        f"nu={nu}, t_end={t_end}",
        f"{'N':>4}  {'rel L2 error vs reference':>28}",
    ]
    errs = []
    for N in Ns:
        grid = SpectralGrid(N, N)
        # same physical initial condition, sampled on this grid: build it on
        # the reference grid then truncate in Fourier space (spectral
        # restriction), so every resolution starts from consistent physics
        omega0 = _restrict(omega0_ref, grid_ref, grid)
        dt = dt_ref  # dt well within the stability limit at every N tested
        p = SimParams(Nx=N, Ny=N, nu=nu, dt=dt, t_end=t_end, save_every=10**9)
        res = run(p, omega0)
        omega_final_hat = grid.fft2(res.omega[-1])

        # compare on the low-resolution grid's resolved wavenumbers only
        ref_on_N = _restrict(res_ref.omega[-1], grid_ref, grid)
        err = float(
            np.linalg.norm(res.omega[-1] - ref_on_N) / np.linalg.norm(ref_on_N)
        )
        errs.append(err)
        lines.append(f"{N:4d}  {err:28.4e}")

    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.semilogy(Ns, errs, "o-", color="#2ca02c")
    ax.set_xlabel("N (grid points per direction)")
    ax.set_ylabel(r"relative $L^2$ error vs. $N=256$ reference")
    ax.set_title("Spatial self-convergence: decaying 2D turbulence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_spatial_turbulence.png", dpi=150)

    return lines


def _restrict(field: np.ndarray, grid_from: SpectralGrid, grid_to: SpectralGrid) -> np.ndarray:
    """Spectral restriction: keep only the low wavenumbers that fit on grid_to.

    Implemented by fftshift-ing to centre the zero mode, cropping a centred
    N_to x N_to block (which keeps exactly the lowest N_to/2 wavenumbers in
    each direction), and ifftshift-ing back before the inverse transform.
    """
    N_from = grid_from.Nx
    N_to = grid_to.Nx
    field_hat = np.fft.fftshift(grid_from.fft2(field))
    lo = N_from // 2 - N_to // 2
    hi = lo + N_to
    cropped = field_hat[lo:hi, lo:hi]
    cropped = np.fft.ifftshift(cropped)
    scale = (N_to * N_to) / (N_from * N_from)
    return np.real(np.fft.ifft2(cropped)) * scale


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    lines_a = mms_resolution_sweep()
    lines_b = turbulence_self_convergence()
    (RESULTS_DIR / "convergence_spatial.txt").write_text("\n".join(lines_a + lines_b) + "\n")
    print(f"\nWrote figures to {FIG_DIR} and results/convergence_spatial.txt")


if __name__ == "__main__":
    main()
