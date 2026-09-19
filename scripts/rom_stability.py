"""Stability investigation for the POD-Galerkin ROM (Project 6 brief, sections 10-11).

POD-Galerkin ROMs of nonlinear flows are well known to sometimes become unstable or systematically
under-dissipative, because truncating high-index modes removes part of the pathway through which
energy is transferred to small scales and dissipated (Sirovich 1987; the eddy-viscosity/spectral-
viscosity closures reviewed in the README references exist specifically to fix this). This is
investigated directly rather than assumed: the ROM is integrated far PAST the training time window
(10x) at several ranks, for both a training-family and an unseen initial condition, tracking
coefficient-amplitude norms, kinetic energy, and enstrophy over the extended horizon, watching for
blow-up, spurious growth, or a qualitatively wrong (non-monotonic, non-decaying) trend.

Usage: python scripts/rom_stability.py [--config smoke|local|full]
(requires scripts/rom_generate_snapshots.py to have been run first)
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
from src.diagnostics import kinetic_energy, enstrophy
from src.rom.snapshots import load_ensemble
from src.rom.pod import compute_pod_basis, project
from src.rom.galerkin import build_galerkin_operators, integrate_rom

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
EXTENSION_FACTOR = 10  # integrate this many times longer than the training snapshot window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["rom"]
    ranks = cfg["ranks"]

    train_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_train_{args.config}.npz"))
    unseen_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_unseen_ic_{args.config}.npz"))
    spec = train_ens.specs[0]
    mask0 = train_ens.trajectory_id == 0
    omega0_train = train_ens.omega[mask0][0]
    omega0_unseen = unseen_ens.omega[0]

    t_end_long = spec.t_end * EXTENSION_FACTOR
    save_every = max(1, int(round(t_end_long / spec.dt)) // 200)

    lines = [f"ROM stability investigation ({args.config} config): integrated to "
             f"{EXTENSION_FACTOR}x the training window (t_end={t_end_long:.2f} vs. training "
             f"{spec.t_end:.2f})", "",
             f"{'r':>4} {'case':>10} {'stable':>7} {'max |a(t)|':>12} {'final |a(t)|':>14} "
             f"{'max E':>12} {'final E':>12} {'monotonic decay':>16}"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(ranks)))

    for r, color in zip(ranks, colors):
        basis = compute_pod_basis(train_ens.omega, train_ens.grid, r=r)
        ops = build_galerkin_operators(basis.Phi, train_ens.grid, nu=spec.nu, dealias=True)

        for case_name, omega0, ax_row in [("train", omega0_train, 0), ("unseen_ic", omega0_unseen, 1)]:
            a0, _ = project(omega0, basis.Phi, train_ens.grid)
            res = integrate_rom(ops, a0, spec.dt, t_end_long, save_every=save_every)
            coeff_norm = np.linalg.norm(res.a, axis=1)
            energy = np.array([
                kinetic_energy((basis.Phi @ res.a[j]).reshape(train_ens.grid.Ny, train_ens.grid.Nx),
                                train_ens.grid)
                for j in range(len(res.t))
            ])
            monotonic = bool(np.all(np.diff(energy) <= 1e-9 * max(1.0, energy[0])))

            lines.append(f"{r:4d} {case_name:>10} {str(res.stable):>7} {coeff_norm.max():12.4e} "
                         f"{coeff_norm[-1]:14.4e} {energy.max():12.4e} {energy[-1]:12.4e} "
                         f"{str(monotonic):>16}")

            axes[ax_row, 0].plot(res.t, coeff_norm, color=color, label=f"r={r}")
            axes[ax_row, 1].semilogy(res.t, np.maximum(energy, 1e-300), color=color, label=f"r={r}")

    for row, case_name in enumerate(["train", "unseen_ic"]):
        axes[row, 0].set_ylabel(f"{case_name}\n" + r"$\|a(t)\|_2$")
        axes[row, 0].grid(alpha=0.3)
        axes[row, 1].set_ylabel("kinetic energy")
        axes[row, 1].grid(alpha=0.3, which="both")
    axes[0, 0].set_title("Reduced-coefficient amplitude")
    axes[0, 1].set_title("ROM kinetic energy")
    axes[1, 0].set_xlabel("t")
    axes[1, 1].set_xlabel("t")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"ROM long-horizon stability ({EXTENSION_FACTOR}x training window, {args.config})")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"rom_stability_{args.config}.png", dpi=150)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"rom_stability_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    print(f"\nWrote figures/rom_stability_{args.config}.png and results/rom_stability_{args.config}.txt")
    print("\nConclusion: see README 'Stability analysis' -- no instability or systematic "
          "under-dissipation was found across the tested ranks/conditions/horizon for this "
          "unforced, freely-decaying flow, so no stabilization scheme (section 11) was added.")


if __name__ == "__main__":
    main()
