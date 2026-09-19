"""Independently verify the precomputed Galerkin reduced operators (Project 6 brief, section 7)
before any ROM trajectory is trusted.

For random reduced states a, compares the fast precomputed-tensor evaluation `reduced_rhs(a, ops)`
against the projected FULL FOM right-hand side (`projected_fom_rhs`): reconstruct omega_r = Phi @ a,
evaluate the FOM's own nonlinear-advection + viscous-Laplacian right-hand side at that field, and
project the result back onto the POD basis. As derived in src/rom/galerkin.py's module docstring,
these two quantities are algebraically IDENTICAL (not merely close for small perturbations), so any
discrepancy above floating-point/FFT round-off indicates a genuine implementation bug.

Usage: python scripts/rom_verify_operators.py [--config smoke|local|full]
(requires scripts/rom_generate_snapshots.py to have been run first)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.rom.snapshots import load_ensemble
from src.rom.pod import compute_pod_basis
from src.rom.galerkin import build_galerkin_operators, verify_operators

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["rom"]

    train_ens = load_ensemble(str(RESULTS_DIR / f"rom_snapshots_train_{args.config}.npz"))

    lines = [f"Galerkin reduced-operator verification ({args.config} config): grid "
             f"{train_ens.grid.Nx}x{train_ens.grid.Ny}", "",
             f"{'r':>4} {'dealias':>8} {'max |tensor - projected FOM RHS|':>34} "
             f"{'typical RHS magnitude':>24} {'relative':>12}"]

    for r in cfg["ranks"]:
        basis = compute_pod_basis(train_ens.omega, train_ens.grid, r=r)
        for dealias in [True, False]:
            ops = build_galerkin_operators(basis.Phi, train_ens.grid, nu=cfg["training"][0]["nu"],
                                            dealias=dealias)
            discrepancies = verify_operators(basis.Phi, ops, train_ens.grid, n_trials=8, seed=0)
            # typical RHS magnitude, for a relative-scale reference
            rng = np.random.default_rng(1)
            a = rng.uniform(-1, 1, r)
            from src.rom.galerkin import reduced_rhs
            typical = float(np.max(np.abs(reduced_rhs(a, ops))))
            rel = float(np.max(discrepancies)) / max(typical, 1e-300)
            lines.append(f"{r:4d} {str(dealias):>8} {np.max(discrepancies):34.6e} "
                         f"{typical:24.6e} {rel:12.6e}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"rom_operator_verification_{args.config}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print(f"\nWrote results/rom_operator_verification_{args.config}.txt")


if __name__ == "__main__":
    main()
