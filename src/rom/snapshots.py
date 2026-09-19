"""Reproducible snapshot generation for POD-Galerkin ROM construction.

Snapshots are drawn from the verified pseudo-spectral FOM (`src.solver.run`), never from a separate
or approximate integrator -- the ROM is built entirely from trusted full-order data. A "snapshot
ensemble" here is a flat stack of vorticity fields, physical space, drawn from one or more FOM
trajectories (a "trajectory family"), together with full metadata (grid resolution, viscosity, time
interval, snapshot spacing, initial-condition parameters, random seeds) so every downstream POD/ROM
result is exactly reproducible and no basis-construction/evaluation leakage goes unnoticed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..grid import SpectralGrid
from ..initial_conditions import random_vorticity_field
from ..params import SimParams
from ..solver import run


@dataclass
class TrajectorySpec:
    """One FOM trajectory's parameters: an initial-condition seed/shape plus a physical/numerical
    configuration. `label` identifies the trajectory in saved metadata (e.g. "train_seed1",
    "unseen_ic", "unseen_nu")."""

    label: str
    N: int
    nu: float
    dt: float
    t_end: float
    k0: float
    seed: int
    n_snapshots: int  # number of saved snapshots (spacing derived to hit this count)
    target_enstrophy: float = 1.0


@dataclass
class SnapshotEnsemble:
    omega: np.ndarray  # (M, Ny, Nx) physical-space vorticity snapshots, all trajectories stacked
    t: np.ndarray  # (M,) simulation time of each snapshot (relative to its own trajectory's t=0)
    trajectory_id: np.ndarray  # (M,) integer index into `specs` identifying the source trajectory
    grid: SpectralGrid = field(repr=False)
    specs: list[TrajectorySpec] = field(default_factory=list)

    @property
    def M(self) -> int:
        return self.omega.shape[0]

    def flat(self) -> np.ndarray:
        """Snapshots as an (N_dof, M) matrix, one flattened field per column -- the conventional
        snapshot-matrix layout for POD/SVD."""
        return self.omega.reshape(self.M, -1).T


def _snapshot_spacing(t_end: float, dt: float, n_snapshots: int) -> int:
    n_steps = int(round(t_end / dt))
    return max(1, n_steps // max(1, n_snapshots - 1))


def generate_trajectory(spec: TrajectorySpec) -> tuple[np.ndarray, np.ndarray, SpectralGrid]:
    """Run one FOM trajectory and return (omega snapshots (n,Ny,Nx), t (n,), grid)."""
    grid = SpectralGrid(spec.N, spec.N)
    omega0 = random_vorticity_field(grid, seed=spec.seed, k0=spec.k0,
                                     target_enstrophy=spec.target_enstrophy)
    save_every = _snapshot_spacing(spec.t_end, spec.dt, spec.n_snapshots)
    params = SimParams(Nx=spec.N, Ny=spec.N, nu=spec.nu, dt=spec.dt, t_end=spec.t_end,
                        save_every=save_every)
    res = run(params, omega0)
    return res.omega, res.t, res.grid


def build_snapshot_ensemble(specs: list[TrajectorySpec]) -> SnapshotEnsemble:
    """Run every trajectory in `specs` and stack their snapshots into one ensemble. All trajectories
    must share the same grid resolution (N) so snapshots live in the same vector space; this is
    checked explicitly rather than silently assumed."""
    if len({s.N for s in specs}) != 1:
        raise ValueError("All trajectories in one snapshot ensemble must share the same grid N "
                          "(POD requires all snapshots to live in the same vector space).")

    omega_list, t_list, traj_id_list = [], [], []
    grid = None
    for i, spec in enumerate(specs):
        omega, t, grid = generate_trajectory(spec)
        omega_list.append(omega)
        t_list.append(t)
        traj_id_list.append(np.full(len(t), i, dtype=int))

    return SnapshotEnsemble(
        omega=np.concatenate(omega_list, axis=0),
        t=np.concatenate(t_list, axis=0),
        trajectory_id=np.concatenate(traj_id_list, axis=0),
        grid=grid,
        specs=specs,
    )


def save_ensemble(ensemble: SnapshotEnsemble, path: str) -> None:
    meta = {
        "specs": [vars(s) for s in ensemble.specs],
        "N": ensemble.grid.Nx,
        "Lx": ensemble.grid.Lx,
        "Ly": ensemble.grid.Ly,
    }
    np.savez_compressed(
        path,
        omega=ensemble.omega,
        t=ensemble.t,
        trajectory_id=ensemble.trajectory_id,
        metadata_json=json.dumps(meta),
    )


def load_ensemble(path: str) -> SnapshotEnsemble:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata_json"]))
    grid = SpectralGrid(meta["N"], meta["N"], meta["Lx"], meta["Ly"])
    specs = [TrajectorySpec(**s) for s in meta["specs"]]
    return SnapshotEnsemble(
        omega=data["omega"], t=data["t"], trajectory_id=data["trajectory_id"],
        grid=grid, specs=specs,
    )
