"""Decaying 2D turbulence: second flow benchmark.

No exact solution exists for this flow; validation here is via physical
diagnostics rather than pointwise comparison: incompressibility, monotonic
energy decay, and the exact energy budget dE/dt = -2*nu*enstrophy that must
hold for any unforced periodic 2D Navier-Stokes flow regardless of the
specific initial condition. See scripts/convergence_spatial.py (Part B) for
the accompanying spatial self-convergence study.

Usage: python examples/decaying_turbulence.py [--config smoke|local|full]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import SimParams, SpectralGrid, run, kinetic_energy, enstrophy, max_divergence
from src.initial_conditions import random_vorticity_field

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())
    turb = cfg["decaying_turbulence"]

    N = turb["N"]
    nu = turb["nu"]
    dt = turb["dt"]
    t_end = turb["t_end"]
    k0 = turb["k0"]
    seed = turb["seed"]
    n_snapshots = turb.get("n_snapshots", 4)

    grid = SpectralGrid(N, N)
    omega0 = random_vorticity_field(grid, seed=seed, k0=k0, target_enstrophy=1.0)

    save_every = max(1, int(round(t_end / dt)) // 200)  # ~200 diagnostic points
    p = SimParams(Nx=N, Ny=N, nu=nu, dt=dt, t_end=t_end, save_every=save_every)
    res = run(p, omega0)

    E = np.array([kinetic_energy(w, grid) for w in res.omega])
    Omega = np.array([enstrophy(w, grid) for w in res.omega])
    max_div = max(max_divergence(w, grid) for w in res.omega[:: max(1, len(res.omega) // 10)])

    dE_dt_numeric = np.gradient(E, res.t)
    dE_dt_theory = -2.0 * nu * Omega
    budget_rel_err = np.max(
        np.abs(dE_dt_numeric[2:-2] - dE_dt_theory[2:-2]) / (np.abs(dE_dt_theory[2:-2]) + 1e-12)
    )

    print(f"Decaying 2D turbulence, N={N}, nu={nu}, k0={k0}, t_end={t_end}")
    print(f"energy: {E[0]:.4f} -> {E[-1]:.4f} (monotonic: {bool(np.all(np.diff(E) <= 1e-12))})")
    print(f"enstrophy: {Omega[0]:.4f} -> {Omega[-1]:.4f}")
    print(f"max |div(u)| over run (sampled): {max_div:.3e}")
    print(f"max relative mismatch of dE/dt vs -2*nu*enstrophy: {budget_rel_err:.3e}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"decaying_turbulence_{args.config}.txt").write_text(
        f"N={N} nu={nu} k0={k0} t_end={t_end} dt={dt} seed={seed}\n"
        f"E(0)={E[0]:.6f} E(t_end)={E[-1]:.6f}\n"
        f"Omega(0)={Omega[0]:.6f} Omega(t_end)={Omega[-1]:.6f}\n"
        f"max|div(u)| (sampled)={max_div:.6e}\n"
        f"max relative energy-budget mismatch={budget_rel_err:.6e}\n"
    )

    # --- figures -----------------------------------------------------
    fig1, axes = plt.subplots(1, n_snapshots, figsize=(3.2 * n_snapshots, 3.4))
    snap_idx = np.linspace(0, len(res.omega) - 1, n_snapshots).astype(int)
    for ax, idx in zip(axes, snap_idx):
        im = ax.pcolormesh(grid.X, grid.Y, res.omega[idx], shading="auto", cmap="RdBu_r")
        ax.set_title(f"t={res.t[idx]:.3f}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig1.suptitle(f"Decaying 2D turbulence, N={N}, Re-like nu={nu}")
    fig1.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig1.savefig(FIG_DIR / f"decaying_turbulence_snapshots_{args.config}.png", dpi=150)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(res.t, E, color="#1f77b4")
    ax1.set_xlabel("t")
    ax1.set_ylabel("kinetic energy E(t)")
    ax1.set_title("Energy decay")
    ax1.grid(alpha=0.3)

    ax2.semilogy(res.t[1:-1], np.abs(dE_dt_numeric[1:-1]), label=r"$|dE/dt|$ (numerical, finite diff.)")
    ax2.semilogy(res.t[1:-1], np.abs(dE_dt_theory[1:-1]), "--", label=r"$2\nu\,\Omega(t)$ (theory)")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"dissipation rate")
    ax2.set_title("Energy budget check")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / f"decaying_turbulence_diagnostics_{args.config}.png", dpi=150)

    print(f"Wrote figures to {FIG_DIR} and results/decaying_turbulence_{args.config}.txt")


if __name__ == "__main__":
    main()
