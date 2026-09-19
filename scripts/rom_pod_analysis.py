"""POD basis construction and the projection-error baseline (Project 6 brief, sections 3-5).

Builds the POD basis from the training snapshot ensemble, reports the singular-value decay and
cumulative captured energy, plots representative modes, and -- crucially -- computes the actual
dynamical projection error (best-possible reconstruction error at each rank, on BOTH the training
data and the held-out unseen trajectories) BEFORE any ROM is integrated. This separates basis
truncation error from ROM dynamical error, which is compared later in
scripts/rom_baseline_evaluation.py: a ROM cannot be expected to do better than this floor.

Usage: python scripts/rom_pod_analysis.py [--config smoke|local|full]
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
from src.rom.snapshots import load_ensemble
from src.rom.pod import compute_pod_basis, cumulative_energy_fraction, projection_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["rom"]
    ranks = cfg["ranks"]
    r_max = max(ranks)

    train_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_train_{args.config}.npz"))
    unseen_ic_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_unseen_ic_{args.config}.npz"))
    unseen_nu_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_unseen_nu_{args.config}.npz"))
    unseen_nu_same_ic_ens = load_ensemble(
        str(RESULTS_DIR / f"rom_snapshots_unseen_nu_same_ic_{args.config}.npz"))

    basis = compute_pod_basis(train_ens.omega, train_ens.grid, r=min(r_max, train_ens.M))
    cum_energy = cumulative_energy_fraction(basis.all_singular_values)

    # --- singular-value decay and cumulative energy ---------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.semilogy(np.arange(1, len(basis.all_singular_values) + 1), basis.all_singular_values, "o-",
                 ms=3)
    ax1.set_xlabel("mode index $i$")
    ax1.set_ylabel(r"singular value $\sigma_i$")
    ax1.set_title("Singular-value decay")
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(np.arange(1, len(cum_energy) + 1), cum_energy, "o-", ms=3)
    for r in ranks:
        if r <= len(cum_energy):
            ax2.axvline(r, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("rank $r$")
    ax2.set_ylabel("cumulative captured energy fraction")
    ax2.set_title("Cumulative POD energy")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"rom_pod_singular_values_{args.config}.png", dpi=150)

    # --- leading POD modes ------------------------------------------------------------------------
    n_show = min(6, basis.Phi.shape[1])
    fig2, axes = plt.subplots(1, n_show, figsize=(2.6 * n_show, 3))
    grid = train_ens.grid
    for i, ax in enumerate(axes):
        im = ax.pcolormesh(grid.X, grid.Y, basis.mode_field(i), shading="auto", cmap="RdBu_r")
        ax.set_title(f"mode {i+1}\n" + r"$\sigma$=" + f"{basis.singular_values[i]:.3g}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle(f"Leading POD modes ({args.config} config, training ensemble)")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / f"rom_pod_modes_{args.config}.png", dpi=150)

    # --- projection-error baseline: actual dynamical reconstruction error vs. rank -----------------
    def mean_proj_error(ensemble, r):
        Phi_r = basis.Phi[:, :r]
        errs = [projection_error(ensemble.omega[j], Phi_r, ensemble.grid)
                for j in range(ensemble.M)]
        return float(np.mean(errs)), np.array(errs)

    lines = [f"POD projection-error baseline ({args.config} config): grid "
             f"{train_ens.grid.Nx}x{train_ens.grid.Ny}, {len(basis.all_singular_values)} total modes",
             "", f"{'r':>4} {'cum. energy':>12} {'train err':>12} "
                 f"{'unseen_ic err':>14} {'unseen_nu err':>14} {'unseen_nu_same_ic err':>22}"]
    err_vs_time = {}
    for r in ranks:
        e_train, err_t_train = mean_proj_error(train_ens, r)
        e_ic, err_t_ic = mean_proj_error(unseen_ic_ens, r)
        e_nu, err_t_nu = mean_proj_error(unseen_nu_ens, r)
        e_nu_same_ic, _ = mean_proj_error(unseen_nu_same_ic_ens, r)
        err_vs_time[r] = (unseen_ic_ens.t, err_t_ic)
        lines.append(f"{r:4d} {cum_energy[r-1]:12.6f} {e_train:12.6e} {e_ic:14.6e} {e_nu:14.6e} "
                      f"{e_nu_same_ic:22.6e}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"rom_pod_projection_error_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    # --- projection error vs. rank (mean over ensemble) ---------------------------------------------
    fig3, ax = plt.subplots(figsize=(5.5, 4.5))
    for label, ens in [("train", train_ens), ("unseen_ic", unseen_ic_ens),
                        ("unseen_nu", unseen_nu_ens),
                        ("unseen_nu_same_ic (control)", unseen_nu_same_ic_ens)]:
        errs = [mean_proj_error(ens, r)[0] for r in ranks]
        ax.semilogy(ranks, errs, "o-", label=label)
    ax.set_xlabel("POD rank $r$")
    ax.set_ylabel("mean relative L2 projection error")
    ax.set_title("Projection-error floor vs. rank")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / f"rom_projection_error_vs_rank_{args.config}.png", dpi=150)

    # --- projection error vs time, unseen_ic, for each rank -----------------------------------------
    fig4, ax = plt.subplots(figsize=(6, 4.5))
    for r in ranks:
        t, errs_t = err_vs_time[r]
        ax.plot(t, errs_t, label=f"r={r}")
    ax.set_xlabel("t")
    ax.set_ylabel("relative L2 projection error")
    ax.set_title("Projection-error floor vs. time (unseen IC)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(FIG_DIR / f"rom_projection_error_vs_time_{args.config}.png", dpi=150)

    print(f"\nWrote figures/rom_pod_singular_values_{args.config}.png, "
          f"rom_pod_modes_{args.config}.png, rom_projection_error_vs_rank_{args.config}.png, "
          f"rom_projection_error_vs_time_{args.config}.png, and "
          f"results/rom_pod_projection_error_{args.config}.txt")


if __name__ == "__main__":
    main()
