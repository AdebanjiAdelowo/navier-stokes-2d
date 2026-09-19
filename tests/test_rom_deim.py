import numpy as np

from src.rom.snapshots import TrajectorySpec, build_snapshot_ensemble
from src.rom.pod import compute_pod_basis
from src.rom.deim import (
    build_nonlinear_snapshots, compute_deim_basis, select_deim_points, build_deim_operators,
    deim_nonlinear_rhs, deim_approximation_error,
)
from src.rom.galerkin import build_galerkin_operators, reduced_rhs, velocity_of, spectral_partials


def _setup(N=48, r=16, n_snapshots=60, dealias=True):
    spec = TrajectorySpec("t", N=N, nu=0.01, dt=1e-3, t_end=0.5, k0=7.0, seed=1,
                           n_snapshots=n_snapshots)
    ens = build_snapshot_ensemble([spec])
    basis = compute_pod_basis(ens.omega, ens.grid, r=r)
    nl_snaps = build_nonlinear_snapshots(ens.omega, ens.grid, dealias=dealias)
    return ens, basis, nl_snaps


def test_deim_points_are_distinct_and_in_range():
    ens, basis, nl_snaps = _setup()
    Psi = compute_deim_basis(nl_snaps, m=10)
    points = select_deim_points(Psi)
    assert len(points) == 10
    assert len(set(points.tolist())) == 10
    assert np.all(points >= 0) and np.all(points < nl_snaps.shape[0])


def test_deim_reconstructs_its_own_training_nonlinear_snapshots_well():
    ens, basis, nl_snaps = _setup()
    for m in [4, 8, 16]:
        Psi = compute_deim_basis(nl_snaps, m)
        points = select_deim_points(Psi)
        errs = deim_approximation_error(Psi, points, nl_snaps)
        assert np.mean(errs) < 1e-4


def test_deim_snapshot_reconstruction_error_decreases_with_rank():
    ens, basis, nl_snaps = _setup()
    means = []
    for m in [2, 4, 8, 16]:
        Psi = compute_deim_basis(nl_snaps, m)
        points = select_deim_points(Psi)
        errs = deim_approximation_error(Psi, points, nl_snaps)
        means.append(np.mean(errs))
    assert np.all(np.diff(means) <= 1e-10)


def test_mode_values_at_deim_points_match_true_field_values():
    """The precomputed (r, m) mode-value arrays used for O(r*m) online DEIM evaluation must exactly
    equal the true field values at those points (a pure indexing/linearity identity, independent of
    DEIM's own approximation quality)."""
    ens, basis, nl_snaps = _setup(r=8)
    Psi = compute_deim_basis(nl_snaps, m=8)
    points = select_deim_points(Psi)
    deim_ops = build_deim_operators(basis.Phi, Psi, points, ens.grid)

    a = basis.coefficients[:, 3]
    omega_r = (basis.Phi @ a).reshape(ens.grid.Ny, ens.grid.Nx)
    u_true, v_true = velocity_of(omega_r, ens.grid)
    fx_true, fy_true = spectral_partials(omega_r, ens.grid)

    assert np.allclose(deim_ops.mode_u_pts.T @ a, u_true.reshape(-1)[points], atol=1e-8)
    assert np.allclose(deim_ops.mode_v_pts.T @ a, v_true.reshape(-1)[points], atol=1e-8)
    assert np.allclose(deim_ops.mode_fx_pts.T @ a, fx_true.reshape(-1)[points], atol=1e-8)
    assert np.allclose(deim_ops.mode_fy_pts.T @ a, fy_true.reshape(-1)[points], atol=1e-8)


def test_deim_matches_undealiased_reduced_rhs_to_high_accuracy():
    """DEIM evaluates the nonlinear term POINTWISE, which is naturally consistent with the
    UNDEALIASED advection (a genuinely pointwise/local quantity), not the dealiased one (a global,
    spectral-filter-dependent quantity that cannot be evaluated at a handful of points without a
    full-grid FFT -- see deim.py module docstring). Verified here against the matching (undealiased)
    reference: error should shrink with DEIM rank m, reaching a tiny value well before m=r."""
    ens, basis, nl_snaps = _setup(r=16, dealias=False)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.01, dealias=False)

    prev_err = np.inf
    for m in [4, 8, 16]:
        Psi = compute_deim_basis(nl_snaps, m)
        points = select_deim_points(Psi)
        deim_ops = build_deim_operators(basis.Phi, Psi, points, ens.grid)

        discs = []
        for j in range(0, nl_snaps.shape[1], 5):
            a = basis.coefficients[:, j]
            exact_nl = reduced_rhs(a, ops) - ops.A_lin @ a
            deim_nl = deim_nonlinear_rhs(a, deim_ops)
            den = np.linalg.norm(exact_nl)
            discs.append(np.linalg.norm(exact_nl - deim_nl) / den if den > 0 else 0.0)
        mean_err = np.mean(discs)
        assert mean_err <= prev_err + 1e-10  # non-increasing with rank
        prev_err = mean_err
    assert prev_err < 1e-6  # at m=r=16, should be essentially exact


def test_dealiasing_mismatch_is_small_at_well_resolved_grid():
    """Documents the finding: the PROJECTED (POD-test-mode) nonlinear term differs between the
    dealiased and undealiased advection evaluations, but this difference is small once the grid is
    well resolved relative to the POD modes' bandwidth (unlike the coarse smoke-config grid, where it
    is much larger -- see README 'Hyper-reduction'). This bounds how much accuracy DEIM (implicitly
    undealiased) sacrifices relative to the trusted (dealiased) tensorized ROM at practical
    resolutions."""
    ens, basis, _ = _setup(N=48, r=8)
    ops_dealias = build_galerkin_operators(basis.Phi, ens.grid, nu=0.01, dealias=True)
    ops_raw = build_galerkin_operators(basis.Phi, ens.grid, nu=0.01, dealias=False)

    a = basis.coefficients[:, 0]
    nl_dealias = reduced_rhs(a, ops_dealias) - ops_dealias.A_lin @ a
    nl_raw = reduced_rhs(a, ops_raw) - ops_raw.A_lin @ a
    rel = np.linalg.norm(nl_dealias - nl_raw) / np.linalg.norm(nl_dealias)
    assert rel < 0.1  # well under the ~17-19% seen at the much coarser smoke-config resolution
