"""Taylor-Green vortex validation run: solution snapshots + pointwise error.

Usage: python examples/taylor_green.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import SimParams, SpectralGrid, run, taylor_green_omega, max_divergence

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    N, nu, k = 64, 0.05, 2
    p = SimParams(Nx=N, Ny=N, nu=nu, dt=1e-3, t_end=0.3, save_every=50)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.0, nu, k=k)

    res = run(p, omega0)
    omega_final = res.omega[-1]
    exact_final = taylor_green_omega(grid.X, grid.Y, res.t[-1], nu, k=k)
    error = omega_final - exact_final
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(exact_final)
    max_div = max_divergence(omega_final, grid)

    print(f"Taylor-Green vortex, N={N}, nu={nu}, k={k}, t_end={p.t_end}")
    print(f"relative L2 error at t_end: {rel_l2:.3e}")
    print(f"max |div(u)| at t_end: {max_div:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im0 = axes[0].pcolormesh(grid.X, grid.Y, omega0, shading="auto", cmap="RdBu_r")
    axes[0].set_title(r"$\omega(x,y,0)$")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(grid.X, grid.Y, omega_final, shading="auto", cmap="RdBu_r")
    axes[1].set_title(rf"$\omega(x,y,t={p.t_end})$, numerical")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(grid.X, grid.Y, error, shading="auto", cmap="PuOr")
    axes[2].set_title(rf"error, rel $L^2$ = {rel_l2:.2e}")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.suptitle("Taylor-Green vortex: numerical vs. exact solution")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "taylor_green_validation.png", dpi=150)
    print(f"Wrote {FIG_DIR / 'taylor_green_validation.png'}")


if __name__ == "__main__":
    main()
