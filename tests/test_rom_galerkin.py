import numpy as np

from src.rom.snapshots import TrajectorySpec, build_snapshot_ensemble
from src.rom.pod import compute_pod_basis, l2_inner_product
from src.rom.galerkin import (
    build_galerkin_operators, reduced_rhs, projected_fom_rhs, verify_operators, integrate_rom,
    integrate_rom_with_nonlinear, laplacian_of, spectral_partials,
)


def _basis(N=20, n_snapshots=25, seed=1, r=6, nu=0.02):
    spec = TrajectorySpec("t", N=N, nu=nu, dt=2e-3, t_end=0.2, k0=5.0, seed=seed,
                           n_snapshots=n_snapshots)
    ens = build_snapshot_ensemble([spec])
    return ens, compute_pod_basis(ens.omega, ens.grid, r=r)


def test_reduced_viscous_operator_is_self_adjoint():
    """L_kj = <laplacian(phi_j), phi_k> should be symmetric: the Laplacian is self-adjoint under the
    L2 inner product on a periodic domain (no boundary terms in integration by parts)."""
    ens, basis = _basis()
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    assert np.allclose(ops.A_lin, ops.A_lin.T, atol=1e-8)


def test_reduced_viscous_operator_matches_direct_inner_product():
    ens, basis = _basis(r=4)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=1.0, dealias=False)  # nu=1 => A_lin = L
    lap0 = laplacian_of(basis.mode_field(0), ens.grid)
    expected_col0 = np.array([l2_inner_product(lap0, basis.mode_field(k), ens.grid) for k in range(4)])
    assert np.allclose(ops.A_lin[:, 0], expected_col0, atol=1e-8)


def test_quadratic_tensor_shape():
    ens, basis = _basis(r=5)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    assert ops.Q.shape == (5, 5, 5)


def test_reduced_rhs_matches_projected_fom_rhs_to_machine_precision():
    """This is the central operator-verification check (project 6 brief section 7): for random
    reduced states, the precomputed-tensor evaluation and the projected full FOM right-hand side
    must agree to numerical precision, since they are algebraically identical (see galerkin.py
    module docstring), not merely close due to truncation."""
    ens, basis = _basis(r=6)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    discrepancies = verify_operators(basis.Phi, ops, ens.grid, n_trials=8, seed=1)
    assert np.max(discrepancies) < 1e-9


def test_reduced_rhs_matches_projected_fom_rhs_without_dealiasing_too():
    ens, basis = _basis(r=5)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=False)
    discrepancies = verify_operators(basis.Phi, ops, ens.grid, n_trials=5, seed=2)
    assert np.max(discrepancies) < 1e-9


def test_zero_state_has_zero_rhs():
    """da/dt = 0 for a = 0 (no forcing, no mean-centering, so the origin is a fixed point)."""
    ens, basis = _basis(r=5)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    rhs0 = reduced_rhs(np.zeros(5), ops)
    assert np.allclose(rhs0, 0.0, atol=1e-12)


def test_rom_integration_is_deterministic():
    ens, basis = _basis(r=4)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    a0 = basis.coefficients[:, 0]
    res1 = integrate_rom(ops, a0, dt=2e-3, t_end=0.1, save_every=5)
    res2 = integrate_rom(ops, a0, dt=2e-3, t_end=0.1, save_every=5)
    assert np.array_equal(res1.a, res2.a)
    assert res1.stable and res2.stable


def test_naive_baseline_matches_tensor_rom_trajectory():
    """The "naive" reconstruct-and-call-FOM-RHS baseline (section 6's explicit intermediate
    baseline) and the exact tensorized ROM are mathematically equivalent evaluations of the SAME
    reduced ODE (see galerkin.py module docstring) -- they should therefore produce essentially
    identical trajectories, differing only by floating-point round-off, despite the naive baseline
    being far more expensive per step (see scripts/rom_performance_benchmark.py)."""
    ens, basis = _basis(r=5)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    a0 = basis.coefficients[:, 0]

    tensor_res = integrate_rom(ops, a0, dt=2e-3, t_end=0.1, save_every=5)

    def naive_nl(a):
        return projected_fom_rhs(a, basis.Phi, ens.grid, ops.nu, ops.dealias) - ops.A_lin @ a
    naive_res = integrate_rom_with_nonlinear(ops.A_lin, naive_nl, a0, dt=2e-3, t_end=0.1, save_every=5)

    assert np.allclose(tensor_res.a, naive_res.a, atol=1e-8, rtol=1e-6)


def test_rom_integration_reduces_to_linear_decay_when_quadratic_term_disabled():
    """With Q forced to zero, da/dt = A_lin @ a is a pure linear ODE; its exact solution is
    a(t) = expm(A_lin*t) @ a0, giving an independent cross-check of the integrator against a
    closed-form (matrix-exponential) reference unrelated to the RK4 stepping logic itself."""
    from scipy.linalg import expm
    ens, basis = _basis(r=4)
    ops = build_galerkin_operators(basis.Phi, ens.grid, nu=0.02, dealias=True)
    ops.Q[:] = 0.0
    a0 = basis.coefficients[:, 0]
    res = integrate_rom(ops, a0, dt=1e-3, t_end=0.05, save_every=50)
    expected_final = expm(ops.A_lin * res.t[-1]) @ a0
    assert np.allclose(res.a[-1], expected_final, rtol=1e-6, atol=1e-8)
