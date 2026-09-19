import numpy as np

from src.diagnostics import kinetic_energy
from src.rom.snapshots import TrajectorySpec, build_snapshot_ensemble
from src.rom.pod import compute_pod_basis, project
from src.rom.galerkin import build_galerkin_operators, integrate_rom


def _setup(r=6):
    specs = [TrajectorySpec("t", N=20, nu=0.02, dt=2e-3, t_end=0.2, k0=5.0, seed=1, n_snapshots=20)]
    ens = build_snapshot_ensemble(specs)
    basis = compute_pod_basis(ens.omega, ens.grid, r=r)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    a0, _ = project(ens.omega[0], basis.Phi, ens.grid)
    return ens, basis, ops, a0


def test_rom_remains_stable_well_past_training_window():
    """Regression guard for the stability finding in scripts/rom_stability.py: integrating far past
    the training window should not blow up for this unforced, freely-decaying flow class."""
    ens, basis, ops, a0 = _setup()
    res = integrate_rom(ops, a0, dt=2e-3, t_end=2.0, save_every=100)  # 10x the training window
    assert res.stable
    assert np.all(np.isfinite(res.a))


def test_rom_kinetic_energy_decays_and_does_not_grow_spuriously():
    ens, basis, ops, a0 = _setup()
    res = integrate_rom(ops, a0, dt=2e-3, t_end=2.0, save_every=50)
    energy = np.array([
        kinetic_energy((basis.Phi @ res.a[j]).reshape(ens.grid.Ny, ens.grid.Nx), ens.grid)
        for j in range(len(res.t))
    ])
    assert energy[-1] < energy[0]
    # energy should not increase by more than a tiny numerical tolerance at any recorded step
    assert np.all(np.diff(energy) <= 1e-10 * max(1.0, energy[0]))
