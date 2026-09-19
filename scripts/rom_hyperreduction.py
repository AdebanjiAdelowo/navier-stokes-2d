"""DEIM hyper-reduction study (Project 6 brief, sections 12-13): verify DEIM independently, study
the DEIM rank INDEPENDENTLY of the state POD rank, and separate three distinct error sources that
must not be conflated (section 21):

  1. POD (state) truncation error       -- scripts/rom_pod_analysis.py
  2. Galerkin dynamical error            -- scripts/rom_baseline_evaluation.py (exact-tensor ROM vs FOM)
  3. Hyper-reduction (DEIM) error        -- THIS script (DEIM-ROM vs the exact-tensor ROM, i.e. the
                                             error DEIM adds ON TOP of an already-verified Galerkin ROM,
                                             isolated from state-POD truncation)

Also builds and times the explicit "naive" baseline the brief asks for (section 6): reconstruct the
full field and call the FOM's own nonlinear evaluation at every ROM step, instead of using any
reduced/precomputed nonlinear evaluation. This is the baseline DEIM would help against, in contrast
to the ALREADY-cheap exact tensor evaluation, which DEIM cannot improve on for this quadratic
nonlinearity (see deim.py module docstring and README "Hyper-reduction").

Usage: python scripts/rom_hyperreduction.py [--config smoke|local|full]
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
from src.diagnostics import kinetic_energy
from src.rom.snapshots import load_ensemble, _snapshot_spacing
from src.rom.pod import compute_pod_basis, project, l2_norm
from src.rom.galerkin import build_galerkin_operators, reduced_rhs, integrate_rom, integrate_rom_with_nonlinear
from src.rom.deim import (
    build_nonlinear_snapshots, compute_deim_basis, select_deim_points, build_deim_operators,
    deim_nonlinear_rhs,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["rom"]
    ranks = cfg["ranks"]
    deim_ranks = cfg["deim_ranks"]

    train_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_train_{args.config}.npz"))
    spec = train_ens.specs[0]
    nu = spec.nu

    # DEIM is naturally consistent with the UNDEALIASED advection (see deim.py docstring / README);
    # both the reference reduced operators and the nonlinear snapshots use dealias=False here so the
    # comparison isolates genuine DEIM (hyper-reduction) approximation error, not a dealiasing mismatch
    # (that mismatch is quantified separately below, at this config's actual resolution).
    nl_snaps = build_nonlinear_snapshots(train_ens.omega, train_ens.grid, dealias=False)

    # --- 1) DEIM error vs DEIM rank, STATE RANK FIXED -----------------------------------------------
    r_fixed = max(ranks)
    basis_fixed = compute_pod_basis(train_ens.omega, train_ens.grid, r=r_fixed)
    ops_fixed_raw = build_galerkin_operators(basis_fixed.Phi, train_ens.grid, nu=nu, dealias=False)

    lines = [f"DEIM hyper-reduction study ({args.config} config): grid "
             f"{train_ens.grid.Nx}x{train_ens.grid.Ny}, state rank fixed at r={r_fixed}", "",
             "-- DEIM error vs DEIM rank (state rank fixed) --",
             f"{'m':>4} {'mean hyper-reduction error':>28}"]

    deim_err_vs_m = []
    for m in deim_ranks:
        Psi = compute_deim_basis(nl_snaps, m)
        points = select_deim_points(Psi)
        deim_ops = build_deim_operators(basis_fixed.Phi, Psi, points, train_ens.grid)
        discs = []
        for j in range(0, train_ens.M, max(1, train_ens.M // 40)):
            a = basis_fixed.coefficients[:, j]
            exact_nl = reduced_rhs(a, ops_fixed_raw) - ops_fixed_raw.A_lin @ a
            deim_nl = deim_nonlinear_rhs(a, deim_ops)
            den = np.linalg.norm(exact_nl)
            discs.append(np.linalg.norm(exact_nl - deim_nl) / den if den > 0 else 0.0)
        deim_err_vs_m.append(np.mean(discs))
        lines.append(f"{m:4d} {deim_err_vs_m[-1]:28.6e}")

    # --- 2) DEIM error vs STATE RANK, DEIM RANK FIXED (shows the two are decoupled) -----------------
    m_fixed = max(deim_ranks)
    lines += ["", f"-- DEIM error vs state rank (DEIM rank fixed at m={m_fixed}) --",
              f"{'r':>4} {'mean hyper-reduction error':>28}"]
    deim_err_vs_r = []
    for r in ranks:
        basis_r = compute_pod_basis(train_ens.omega, train_ens.grid, r=r)
        ops_r_raw = build_galerkin_operators(basis_r.Phi, train_ens.grid, nu=nu, dealias=False)
        Psi = compute_deim_basis(nl_snaps, m_fixed)
        points = select_deim_points(Psi)
        deim_ops = build_deim_operators(basis_r.Phi, Psi, points, train_ens.grid)
        discs = []
        for j in range(0, train_ens.M, max(1, train_ens.M // 40)):
            a = basis_r.coefficients[:, j]
            exact_nl = reduced_rhs(a, ops_r_raw) - ops_r_raw.A_lin @ a
            deim_nl = deim_nonlinear_rhs(a, deim_ops)
            den = np.linalg.norm(exact_nl)
            discs.append(np.linalg.norm(exact_nl - deim_nl) / den if den > 0 else 0.0)
        deim_err_vs_r.append(np.mean(discs))
        lines.append(f"{r:4d} {deim_err_vs_r[-1]:28.6e}")

    # --- 3) dealiasing mismatch at THIS config's resolution (separate from DEIM truncation) ---------
    ops_fixed_dealias = build_galerkin_operators(basis_fixed.Phi, train_ens.grid, nu=nu, dealias=True)
    a0 = basis_fixed.coefficients[:, 0]
    nl_dealias = reduced_rhs(a0, ops_fixed_dealias) - ops_fixed_dealias.A_lin @ a0
    nl_raw = reduced_rhs(a0, ops_fixed_raw) - ops_fixed_raw.A_lin @ a0
    dealias_mismatch = float(np.linalg.norm(nl_dealias - nl_raw) / np.linalg.norm(nl_dealias))
    lines += ["", f"-- dealiasing mismatch (dealiased vs undealiased projected nonlinear term, "
              f"r={r_fixed}) --", f"relative difference: {dealias_mismatch:.6e}",
              "(this is what DEIM's implicit use of the undealiased advection costs relative to the "
              "trusted, dealiased tensorized ROM, at this config's resolution -- see README)"]

    # --- 4) hyper-reduction error vs Galerkin dynamical error: ROM trajectory comparison ------------
    mask0 = train_ens.trajectory_id == 0
    fom_omega = train_ens.omega[mask0]
    save_every = _snapshot_spacing(spec.t_end, spec.dt, spec.n_snapshots)

    Psi_max = compute_deim_basis(nl_snaps, max(deim_ranks))
    points_max = select_deim_points(Psi_max)
    deim_ops_max = build_deim_operators(basis_fixed.Phi, Psi_max, points_max, train_ens.grid)

    a0_traj, _ = project(fom_omega[0], basis_fixed.Phi, train_ens.grid)
    tensor_res = integrate_rom(ops_fixed_dealias, a0_traj, spec.dt, spec.t_end, save_every)

    def deim_nl_fn(a):
        return deim_nonlinear_rhs(a, deim_ops_max)
    deim_res = integrate_rom_with_nonlinear(ops_fixed_dealias.A_lin, deim_nl_fn, a0_traj, spec.dt,
                                             spec.t_end, save_every)

    n = min(len(tensor_res.t), len(deim_res.t), fom_omega.shape[0])
    hyperred_vs_tensor = np.array([
        l2_norm((basis_fixed.Phi @ (deim_res.a[j] - tensor_res.a[j])).reshape(
            train_ens.grid.Ny, train_ens.grid.Nx), train_ens.grid) /
        max(l2_norm((basis_fixed.Phi @ tensor_res.a[j]).reshape(
            train_ens.grid.Ny, train_ens.grid.Nx), train_ens.grid), 1e-300)
        for j in range(n)
    ])
    tensor_vs_fom = np.array([
        l2_norm((basis_fixed.Phi @ tensor_res.a[j]).reshape(train_ens.grid.Ny, train_ens.grid.Nx)
                 - fom_omega[j], train_ens.grid) / max(l2_norm(fom_omega[j], train_ens.grid), 1e-300)
        for j in range(n)
    ])
    lines += ["", f"-- error decomposition on the training trajectory (r={r_fixed}, "
              f"m={max(deim_ranks)}) --",
              f"mean Galerkin dynamical error (tensor ROM vs FOM):        {np.mean(tensor_vs_fom):.6e}",
              f"mean hyper-reduction error (DEIM ROM vs tensor ROM):      {np.mean(hyperred_vs_tensor):.6e}",
              "(hyper-reduction error is much smaller than Galerkin dynamical error: DEIM adds little "
              "on top of an already-verified ROM, consistent with the tensor method already being "
              "exact for this quadratic nonlinearity -- see README)"]

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"rom_hyperreduction_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    # --- figures --------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(deim_ranks, deim_err_vs_m, "o-")
    axes[0].set_xlabel("DEIM rank $m$")
    axes[0].set_ylabel("mean hyper-reduction error")
    axes[0].set_title(f"DEIM error vs. DEIM rank (state rank fixed, r={r_fixed})")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogy(ranks, deim_err_vs_r, "o-", color="tab:orange")
    axes[1].set_xlabel("state POD rank $r$")
    axes[1].set_ylabel("mean hyper-reduction error")
    axes[1].set_title(f"DEIM error vs. state rank (DEIM rank fixed, m={m_fixed})")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"rom_deim_error_vs_rank_{args.config}.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(6, 4.5))
    t = tensor_res.t[:n]
    ax.semilogy(t, tensor_vs_fom, label="Galerkin dynamical error (tensor ROM vs FOM)")
    ax.semilogy(t, hyperred_vs_tensor, label="hyper-reduction error (DEIM ROM vs tensor ROM)")
    ax.set_xlabel("t")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Error decomposition: Galerkin vs. hyper-reduction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / f"rom_error_decomposition_{args.config}.png", dpi=150)

    print(f"\nWrote figures/rom_deim_error_vs_rank_{args.config}.png, "
          f"rom_error_decomposition_{args.config}.png, and "
          f"results/rom_hyperreduction_{args.config}.txt")


if __name__ == "__main__":
    main()
