"""Accuracy-speed trade-off and offline/online cost breakdown (Project 6 brief, sections 14-15) --
one of this project's central results.

Benchmarks FOUR ways of advancing the (reduced or full) state by one time step, all using the SAME
warm-up/repeat methodology and the SAME physical time interval as scripts/benchmark_runtime.py:

  - FOM:              the verified pseudo-spectral solver itself (src.solver.run).
  - tensor ROM:        the exact precomputed-tensor Galerkin evaluation (O(r^3), no grid operations).
  - naive baseline:    reconstruct the full field and call the FOM's own nonlinear RHS at every ROM
                        step (`galerkin.projected_fom_rhs`) -- the explicit intermediate baseline
                        the project brief asks for (section 6), NOT the recommended approach.
  - DEIM-hyper-reduced: reconstruct only at m << N selected points using precomputed mode values
                        (`deim.deim_nonlinear_rhs`), O(r*m), no grid operations.

Offline costs (snapshot generation, POD SVD, Galerkin tensor construction, DEIM basis/point
selection) are timed and reported SEPARATELY from online (per-step / per-simulated-time-unit) cost --
conflating the two would make any "speed-up" number meaningless (section 14).

Usage: python scripts/rom_performance_benchmark.py [--config smoke|local|full]
(requires scripts/rom_generate_snapshots.py to have been run first)
"""
from __future__ import annotations

import argparse
import pathlib
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import SimParams, run
from src.rom.snapshots import load_ensemble, _snapshot_spacing
from src.rom.pod import compute_pod_basis, project, l2_norm
from src.rom.galerkin import (
    build_galerkin_operators, integrate_rom, integrate_rom_with_nonlinear, projected_fom_rhs,
)
from src.rom.deim import (
    build_nonlinear_snapshots, compute_deim_basis, select_deim_points, build_deim_operators,
    deim_nonlinear_rhs,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def get_hardware_info() -> str:
    import subprocess
    cpu = "unknown"
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return (f"Platform: {platform.platform()}\nCPU: {cpu}\nPython: {platform.python_version()}\n"
            f"NumPy: {np.__version__}\nPrecision: float64/complex128, CPU only, single-threaded\n")


def time_repeats(fn, n_warmup: int, n_repeats: int) -> tuple[float, float, float]:
    fn()  # warm-up (discarded)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["rom"]
    ranks = cfg["ranks"]

    train_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_train_{args.config}.npz"))
    spec = train_ens.specs[0]
    grid = train_ens.grid
    N, nu, dt = spec.N, spec.nu, spec.dt

    n_warmup_steps, n_timed_steps, n_repeats = 5, 30, 3
    t_timed = n_timed_steps * dt

    lines = [get_hardware_info(),
             f"grid N={N}, nu={nu}, dt={dt}, warm-up steps={n_warmup_steps}, "
             f"timed steps/repeat={n_timed_steps}, repeats={n_repeats}", ""]

    # --- offline cost (one-time, r = max rank) -------------------------------------------------
    r_max = max(ranks)
    m_max = max(cfg["deim_ranks"])

    t0 = time.perf_counter()
    basis = compute_pod_basis(train_ens.omega, grid, r=r_max)
    t_pod = time.perf_counter() - t0

    t0 = time.perf_counter()
    ops = build_galerkin_operators(basis.Phi, grid, nu=nu, dealias=True)
    t_tensor = time.perf_counter() - t0

    t0 = time.perf_counter()
    nl_snaps = build_nonlinear_snapshots(train_ens.omega, grid, dealias=False)
    t_nlsnap = time.perf_counter() - t0

    t0 = time.perf_counter()
    Psi = compute_deim_basis(nl_snaps, m_max)
    points = select_deim_points(Psi)
    deim_ops = build_deim_operators(basis.Phi, Psi, points, grid)
    t_deim = time.perf_counter() - t0

    lines += ["-- offline cost (one-time, r={}, m={}) --".format(r_max, m_max),
              f"POD SVD (state basis):              {t_pod:8.4f} s",
              f"Galerkin tensor construction:        {t_tensor:8.4f} s",
              f"nonlinear-snapshot construction:      {t_nlsnap:8.4f} s",
              f"DEIM basis + point selection:        {t_deim:8.4f} s",
              f"total offline (excl. FOM snapshot generation, timed separately in "
              f"scripts/rom_generate_snapshots.py): {t_pod + t_tensor + t_nlsnap + t_deim:8.4f} s", ""]

    # --- online cost: ms per simulated time unit, each method -----------------------------------
    omega0 = train_ens.omega[train_ens.trajectory_id == 0][0]
    a0, _ = project(omega0, basis.Phi, grid)

    def fom_call():
        p = SimParams(Nx=N, Ny=N, nu=nu, dt=dt, t_end=t_timed, save_every=10 ** 9)
        run(p, omega0)

    def tensor_call():
        integrate_rom(ops, a0, dt, t_timed, save_every=10 ** 9)

    def naive_call():
        def nl(a):
            return projected_fom_rhs(a, basis.Phi, grid, nu, dealias=True) - ops.A_lin @ a
        integrate_rom_with_nonlinear(ops.A_lin, nl, a0, dt, t_timed, save_every=10 ** 9)

    def deim_call():
        def nl(a):
            return deim_nonlinear_rhs(a, deim_ops)
        integrate_rom_with_nonlinear(ops.A_lin, nl, a0, dt, t_timed, save_every=10 ** 9)

    lines += [f"-- online cost: median/min/max ms per {t_timed:.4f} simulated time units "
              f"({n_timed_steps} steps), r={r_max}, m={m_max} --",
              f"{'method':>16} {'median(ms)':>12} {'min(ms)':>10} {'max(ms)':>10} {'speed-up vs FOM':>16}"]

    results = {}
    for name, fn in [("FOM", fom_call), ("tensor ROM", tensor_call), ("naive baseline", naive_call),
                      ("DEIM ROM", deim_call)]:
        med, mn, mx = time_repeats(fn, n_warmup_steps, n_repeats)
        results[name] = med
        lines.append(f"{name:>16} {med*1000:12.4f} {mn*1000:10.4f} {mx*1000:10.4f} "
                      f"{results['FOM']/med:16.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"rom_performance_benchmark_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    # --- accuracy-vs-speed, across ranks (tensor ROM only, the recommended method) --------------
    mask0 = train_ens.trajectory_id == 0
    fom_omega = train_ens.omega[mask0]
    save_every = _snapshot_spacing(spec.t_end, spec.dt, spec.n_snapshots)

    acc_speed_lines = ["", "-- accuracy vs. speed-up across state rank (tensor ROM, full training "
                       "trajectory) --", f"{'r':>4} {'mean state err':>16} {'online speed-up vs FOM':>24}"]
    errs, speedups = [], []
    for r in ranks:
        basis_r = compute_pod_basis(train_ens.omega, grid, r=r)
        ops_r = build_galerkin_operators(basis_r.Phi, grid, nu=nu, dealias=True)
        a0_r, _ = project(fom_omega[0], basis_r.Phi, grid)

        def call_r():
            integrate_rom(ops_r, a0_r, dt, t_timed, save_every=10 ** 9)
        med_r, _, _ = time_repeats(call_r, n_warmup_steps, n_repeats)
        speedup_r = results["FOM"] / med_r

        res = integrate_rom(ops_r, a0_r, spec.dt, spec.t_end, save_every)
        n = min(len(res.t), fom_omega.shape[0])
        state_errs = [l2_norm((basis_r.Phi @ res.a[j]).reshape(grid.Ny, grid.Nx) - fom_omega[j], grid)
                      / max(l2_norm(fom_omega[j], grid), 1e-300) for j in range(n)]
        errs.append(float(np.mean(state_errs)))
        speedups.append(speedup_r)
        acc_speed_lines.append(f"{r:4d} {errs[-1]:16.6e} {speedup_r:24.2f}")

    (RESULTS_DIR / f"rom_performance_benchmark_{args.config}.txt").write_text(
        "\n".join(lines + acc_speed_lines) + "\n")
    for line in acc_speed_lines:
        print(line)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(speedups, errs, c=ranks, cmap="viridis", s=80, zorder=3)
    for r, sp, e in zip(ranks, speedups, errs):
        ax.annotate(f"r={r}", (sp, e), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("online speed-up vs. FOM (x)")
    ax.set_ylabel("mean relative L2 state error (training trajectory)")
    ax.set_yscale("log")
    ax.set_title(f"Accuracy-speed trade-off ({args.config} config)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"rom_accuracy_speed_tradeoff_{args.config}.png", dpi=150)

    print(f"\nWrote figures/rom_accuracy_speed_tradeoff_{args.config}.png and "
          f"results/rom_performance_benchmark_{args.config}.txt")


if __name__ == "__main__":
    main()
