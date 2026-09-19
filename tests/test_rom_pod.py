import numpy as np

from src.grid import SpectralGrid
from src.rom.snapshots import TrajectorySpec, build_snapshot_ensemble
from src.rom.pod import (
    cell_area, l2_inner_product, l2_norm, compute_pod_basis, cumulative_energy_fraction,
    project, projection_error,
)


def _ensemble(N=20, n_snapshots=25, seed=1):
    spec = TrajectorySpec("t", N=N, nu=0.02, dt=2e-3, t_end=0.2, k0=5.0, seed=seed,
                           n_snapshots=n_snapshots)
    return build_snapshot_ensemble([spec])


def test_l2_inner_product_matches_enstrophy_convention():
    """The discrete L2 inner product used here must match the weighting already used by
    src.diagnostics.enstrophy (0.5 * sum(omega^2) * cell_area), i.e. <f,f> = 2*enstrophy(f)."""
    grid = SpectralGrid(16, 16)
    rng = np.random.default_rng(0)
    f = rng.standard_normal((16, 16))
    from src.diagnostics import enstrophy
    assert np.isclose(l2_inner_product(f, f, grid), 2.0 * enstrophy(f, grid))


def test_l2_inner_product_uses_cell_area_not_raw_euclidean():
    grid = SpectralGrid(8, 8, Lx=2.0, Ly=2.0)
    f = np.ones((8, 8))
    expected = cell_area(grid) * np.sum(f * f)
    assert np.isclose(l2_inner_product(f, f, grid), expected)
    assert not np.isclose(l2_inner_product(f, f, grid), np.sum(f * f))  # NOT raw Euclidean


def test_pod_modes_are_l2_orthonormal():
    ens = _ensemble()
    basis = compute_pod_basis(ens.omega, ens.grid, r=6)
    r = basis.Phi.shape[1]
    G = np.array([[l2_inner_product(basis.mode_field(i), basis.mode_field(j), ens.grid)
                   for j in range(r)] for i in range(r)])
    assert np.allclose(G, np.eye(r), atol=1e-9)


def test_singular_values_are_sorted_descending():
    ens = _ensemble()
    basis = compute_pod_basis(ens.omega, ens.grid, r=8)
    assert np.all(np.diff(basis.singular_values) <= 1e-12)


def test_cumulative_energy_fraction_is_monotonic_and_ends_at_one():
    ens = _ensemble()
    basis = compute_pod_basis(ens.omega, ens.grid, r=ens.M)
    cum = cumulative_energy_fraction(basis.all_singular_values)
    assert np.all(np.diff(cum) >= -1e-12)
    assert np.isclose(cum[-1], 1.0, atol=1e-8)


def test_full_rank_projection_reconstructs_training_snapshot_exactly():
    ens = _ensemble(n_snapshots=15)
    basis = compute_pod_basis(ens.omega, ens.grid, r=ens.M)
    err = projection_error(ens.omega[5], basis.Phi, ens.grid)
    assert err < 1e-10


def test_projection_error_decreases_with_rank():
    ens = _ensemble(n_snapshots=20)
    basis_full = compute_pod_basis(ens.omega, ens.grid, r=ens.M)
    field = ens.omega[10]
    errs = [projection_error(field, basis_full.Phi[:, :r], ens.grid) for r in [1, 3, 6, 10]]
    assert np.all(np.diff(errs) <= 1e-10)


def test_projection_coefficients_match_direct_inner_product():
    ens = _ensemble()
    basis = compute_pod_basis(ens.omega, ens.grid, r=5)
    a_direct, _ = project(ens.omega[0], basis.Phi, ens.grid)
    assert np.allclose(a_direct, basis.coefficients[:, 0], atol=1e-8)


def test_reconstructed_enstrophy_from_full_basis_matches_direct_calculation():
    """Absolute-quantity cross-check exercising the cell_area rescaling explicitly (see pod.py
    module docstring): at full rank, sum_i a_i(t)^2 must equal ||omega(t)||_L2^2 = 2*enstrophy(t)."""
    ens = _ensemble(n_snapshots=15)
    basis = compute_pod_basis(ens.omega, ens.grid, r=ens.M)
    from src.diagnostics import enstrophy
    j = 3
    a_j = basis.coefficients[:, j]
    assert np.isclose(np.sum(a_j ** 2), 2.0 * enstrophy(ens.omega[j], ens.grid), rtol=1e-6)
