"""Generate and save the training and unseen-evaluation snapshot ensembles used by the rest of the
Project 6 ROM pipeline. All downstream ROM scripts read the saved .npz files produced here rather than
regenerating trajectories themselves, so the exact same FOM data is used for POD construction and for
every subsequent evaluation.

Produces three ensembles:
- `rom_snapshots_train_{config}.npz`: the training family (one or more FOM trajectories), used to
  build the POD basis and the Galerkin operators.
- `rom_snapshots_unseen_ic_{config}.npz`: a different initial-condition seed, same physical
  parameters as training -- "in-distribution" in the parameter sense, but not used for basis
  construction.
- `rom_snapshots_unseen_nu_{config}.npz`: a different viscosity, testing whether the ROM built at one
  physical parameter transfers to another.

Usage: python scripts/rom_generate_snapshots.py [--config smoke|local|full]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.rom.snapshots import TrajectorySpec, build_snapshot_ensemble, save_ensemble

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["smoke", "local", "full"])
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / f"{args.config}.yaml").read_text())["rom"]
    RESULTS_DIR.mkdir(exist_ok=True)

    train_specs = [TrajectorySpec(**d) for d in cfg["training"]]
    train_ens = build_snapshot_ensemble(train_specs)
    save_ensemble(train_ens, str(RESULTS_DIR / f"rom_snapshots_train_{args.config}.npz"))
    print(f"train: {len(train_specs)} trajectories, {train_ens.M} snapshots, "
          f"grid {train_ens.grid.Nx}x{train_ens.grid.Ny}")

    unseen_ic_ens = build_snapshot_ensemble([TrajectorySpec(**cfg["unseen_ic"])])
    save_ensemble(unseen_ic_ens, str(RESULTS_DIR / f"rom_snapshots_unseen_ic_{args.config}.npz"))
    print(f"unseen_ic: {unseen_ic_ens.M} snapshots "
          f"(seed={cfg['unseen_ic']['seed']}, nu={cfg['unseen_ic']['nu']})")

    unseen_nu_ens = build_snapshot_ensemble([TrajectorySpec(**cfg["unseen_nu"])])
    save_ensemble(unseen_nu_ens, str(RESULTS_DIR / f"rom_snapshots_unseen_nu_{args.config}.npz"))
    print(f"unseen_nu: {unseen_nu_ens.M} snapshots "
          f"(seed={cfg['unseen_nu']['seed']}, nu={cfg['unseen_nu']['nu']})")

    unseen_nu_same_ic_ens = build_snapshot_ensemble([TrajectorySpec(**cfg["unseen_nu_same_ic"])])
    save_ensemble(unseen_nu_same_ic_ens,
                  str(RESULTS_DIR / f"rom_snapshots_unseen_nu_same_ic_{args.config}.npz"))
    print(f"unseen_nu_same_ic (control): {unseen_nu_same_ic_ens.M} snapshots "
          f"(seed={cfg['unseen_nu_same_ic']['seed']}, nu={cfg['unseen_nu_same_ic']['nu']})")

    print(f"\nWrote results/rom_snapshots_{{train,unseen_ic,unseen_nu,unseen_nu_same_ic}}_"
          f"{args.config}.npz")


if __name__ == "__main__":
    main()
