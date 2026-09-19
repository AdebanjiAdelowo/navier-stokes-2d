"""Baseline POD-Galerkin ROM evaluation across POD rank (Project 6 brief, sections 8-9).

For each rank r, builds the POD basis + reduced operators from the training ensemble, then integrates
the ROM starting from four different initial conditions:

  - train:              a training trajectory's own initial condition (in-sample basis, in-sample
                         dynamics -- tests whether the ROM reproduces the trajectory it was built from)
  - unseen_ic:           a new random-phase realization, same nu as training
  - unseen_nu:           a new random-phase realization, DIFFERENT nu
  - unseen_nu_same_ic:   control -- the training trajectory's own phase, but at the unseen_nu value
                         (isolates the parameter-shift effect from the phase-mismatch effect)

and compares the resulting ROM state/energy/enstrophy trajectories against the FOM's own trajectory,
AND against the POD projection-error floor computed in scripts/rom_pod_analysis.py (a ROM cannot be
expected to do better than that floor; see src/rom/pod.py).

Usage: python scripts/rom_baseline_evaluation.py [--config smoke|local|full]
(requires scripts/rom_generate_snapshots.py to have been run first)
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.diagnostics import kinetic_energy, enstrophy
from src.rom.snapshots import TrajectorySpec, load_ensemble, _snapshot_spacing
from src.rom.pod import compute_pod_basis, project, projection_error, l2_norm
from src.rom.galerkin import build_galerkin_operators, integrate_rom

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def evaluate_on_trajectory(spec: TrajectorySpec, fom_omega, fom_t, basis, ops, grid):
    save_every = _snapshot_spacing(spec.t_end, spec.dt, spec.n_snapshots)
    a0, _ = project(fom_omega[0], basis.Phi, grid)

    t0 = time.perf_counter()
    rom_res = integrate_rom(ops, a0, spec.dt, spec.t_end, save_every=save_every)
    runtime = time.perf_counter() - t0

    n = min(len(rom_res.t), fom_omega.shape[0])
    state_err, energy_err, enstrophy_err, proj_err = [], [], [], []
    for j in range(n):
        omega_rom = (basis.Phi @ rom_res.a[j]).reshape(grid.Ny, grid.Nx)
        omega_fom = fom_omega[j]
        den = l2_norm(omega_fom, grid)
        state_err.append(l2_norm(omega_rom - omega_fom, grid) / den if den > 0 else 0.0)
        proj_err.append(projection_error(omega_fom, basis.Phi, grid))
        E_rom, E_fom = kinetic_energy(omega_rom, grid), kinetic_energy(omega_fom, grid)
        energy_err.append(abs(E_rom - E_fom) / max(abs(E_fom), 1e-300))
        Om_rom, Om_fom = enstrophy(omega_rom, grid), enstrophy(omega_fom, grid)
        enstrophy_err.append(abs(Om_rom - Om_fom) / max(abs(Om_fom), 1e-300))

    return {
        "t": rom_res.t[:n], "state_err": np.array(state_err), "proj_err": np.array(proj_err),
        "energy_err": np.array(energy_err), "enstrophy_err": np.array(enstrophy_err),
        "stable": rom_res.stable, "runtime": runtime, "a": rom_res.a, "n_matched": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["rom"]
    ranks = cfg["ranks"]

    train_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_train_{args.config}.npz"))
    unseen_ic_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_unseen_ic_{args.config}.npz"))
    unseen_nu_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_unseen_nu_{args.config}.npz"))
    unseen_nu_same_ic_ens = load_ensemble(
        str(RESULTS_DIR / f"rom_snapshots_unseen_nu_same_ic_{args.config}.npz"))

    train_spec = train_ens.specs[0]
    train0_mask = train_ens.trajectory_id == 0
    cases = {
        "train": (train_spec, train_ens.omega[train0_mask], train_ens.t[train0_mask], train_spec.nu),
        "unseen_ic": (unseen_ic_ens.specs[0], unseen_ic_ens.omega, unseen_ic_ens.t,
                      unseen_ic_ens.specs[0].nu),
        "unseen_nu": (unseen_nu_ens.specs[0], unseen_nu_ens.omega, unseen_nu_ens.t,
                      unseen_nu_ens.specs[0].nu),
        "unseen_nu_same_ic": (unseen_nu_same_ic_ens.specs[0], unseen_nu_same_ic_ens.omega,
                              unseen_nu_same_ic_ens.t, unseen_nu_same_ic_ens.specs[0].nu),
    }

    results_table = ["Baseline POD-Galerkin ROM evaluation ({}): grid {}x{}".format(
        args.config, train_ens.grid.Nx, train_ens.grid.Ny), "",
        f"{'r':>4} {'case':>18} {'stable':>7} {'mean state err':>15} {'final state err':>16} "
        f"{'mean E err':>11} {'mean Omega err':>15} {'runtime(s)':>11}"]

    all_case_results = {r: {} for r in ranks}
    for r in ranks:
        basis = compute_pod_basis(train_ens.omega, train_ens.grid, r=r)
        for case_name, (spec, fom_omega, fom_t, nu) in cases.items():
            ops = build_galerkin_operators(basis.Phi, train_ens.grid, nu=nu, dealias=True)
            res = evaluate_on_trajectory(spec, fom_omega, fom_t, basis, ops, train_ens.grid)
            all_case_results[r][case_name] = res
            results_table.append(
                f"{r:4d} {case_name:>18} {str(res['stable']):>7} "
                f"{np.mean(res['state_err']):15.6e} {res['state_err'][-1]:16.6e} "
                f"{np.mean(res['energy_err']):11.6e} {np.mean(res['enstrophy_err']):15.6e} "
                f"{res['runtime']:11.4f}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"rom_baseline_evaluation_{args.config}.txt").write_text(
        "\n".join(results_table) + "\n")
    for line in results_table:
        print(line)

    # --- state error vs rank, all cases -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for case_name in cases:
        errs = [np.mean(all_case_results[r][case_name]["state_err"]) for r in ranks]
        ax.semilogy(ranks, errs, "o-", label=case_name)
    ax.set_xlabel("POD rank $r$")
    ax.set_ylabel("mean relative L2 state error (ROM vs FOM)")
    ax.set_title("ROM state error vs. rank")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"rom_state_error_vs_rank_{args.config}.png", dpi=150)

    # --- error vs time: ROM error vs projection floor, largest rank, train + unseen_ic -------------
    r_max = max(ranks)
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, case_name in zip(axes, ["train", "unseen_ic"]):
        res = all_case_results[r_max][case_name]
        ax.semilogy(res["t"], res["state_err"], label="ROM state error")
        ax.semilogy(res["t"], res["proj_err"], "--", label="POD projection-error floor")
        ax.set_xlabel("t")
        ax.set_title(f"{case_name} (r={r_max})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("relative L2 error")
    fig2.suptitle("ROM dynamical error vs. its own projection-error floor")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / f"rom_error_vs_time_{args.config}.png", dpi=150)

    # --- kinetic energy & enstrophy comparison, train case, largest rank ---------------------------
    basis_max = compute_pod_basis(train_ens.omega, train_ens.grid, r=r_max)
    res_train = all_case_results[r_max]["train"]
    E_fom = [kinetic_energy(cases["train"][1][j], train_ens.grid) for j in range(res_train["n_matched"])]
    Om_fom = [enstrophy(cases["train"][1][j], train_ens.grid) for j in range(res_train["n_matched"])]
    E_rom = [kinetic_energy((basis_max.Phi @ res_train["a"][j]).reshape(train_ens.grid.Ny, train_ens.grid.Nx),
                             train_ens.grid) for j in range(res_train["n_matched"])]
    Om_rom = [enstrophy((basis_max.Phi @ res_train["a"][j]).reshape(train_ens.grid.Ny, train_ens.grid.Nx),
                         train_ens.grid) for j in range(res_train["n_matched"])]

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(res_train["t"], E_fom, label="FOM")
    ax1.plot(res_train["t"], E_rom, "--", label=f"ROM (r={r_max})")
    ax1.set_xlabel("t")
    ax1.set_ylabel("kinetic energy")
    ax1.set_title("Kinetic energy: FOM vs ROM (train case)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(res_train["t"], Om_fom, label="FOM")
    ax2.plot(res_train["t"], Om_rom, "--", label=f"ROM (r={r_max})")
    ax2.set_xlabel("t")
    ax2.set_ylabel("enstrophy")
    ax2.set_title("Enstrophy: FOM vs ROM (train case)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / f"rom_energy_enstrophy_comparison_{args.config}.png", dpi=150)

    # --- FOM/ROM vorticity snapshot comparison, train case, largest rank ---------------------------
    n_show = min(4, res_train["n_matched"])
    idx = np.linspace(0, res_train["n_matched"] - 1, n_show).astype(int)
    fig4, axes4 = plt.subplots(2, n_show, figsize=(2.6 * n_show, 5.2))
    grid = train_ens.grid
    for col, j in enumerate(idx):
        omega_fom = cases["train"][1][j]
        omega_rom = (basis_max.Phi @ res_train["a"][j]).reshape(grid.Ny, grid.Nx)
        vmax = max(np.abs(omega_fom).max(), 1e-12)
        axes4[0, col].pcolormesh(grid.X, grid.Y, omega_fom, shading="auto", cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax)
        axes4[0, col].set_title(f"t={res_train['t'][j]:.2f}")
        axes4[1, col].pcolormesh(grid.X, grid.Y, omega_rom, shading="auto", cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax)
        for row in range(2):
            axes4[row, col].set_xticks([])
            axes4[row, col].set_yticks([])
            axes4[row, col].set_aspect("equal")
    axes4[0, 0].set_ylabel("FOM")
    axes4[1, 0].set_ylabel(f"ROM (r={r_max})")
    fig4.suptitle(f"FOM vs ROM vorticity snapshots (train case, {args.config} config)")
    fig4.tight_layout()
    fig4.savefig(FIG_DIR / f"rom_vorticity_snapshots_{args.config}.png", dpi=150)

    print(f"\nWrote figures/rom_state_error_vs_rank_{args.config}.png, "
          f"rom_error_vs_time_{args.config}.png, rom_energy_enstrophy_comparison_{args.config}.png, "
          f"rom_vorticity_snapshots_{args.config}.png, and "
          f"results/rom_baseline_evaluation_{args.config}.txt")


if __name__ == "__main__":
    main()
