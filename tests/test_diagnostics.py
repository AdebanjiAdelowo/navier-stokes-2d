"""Physical diagnostic checks: energy budget, divergence, positivity of enstrophy."""
import numpy as np

from src import SimParams, SpectralGrid, run, kinetic_energy, enstrophy, max_divergence


def _lowpass_random_field(grid: SpectralGrid, seed: int, sigma: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((grid.Ny, grid.Nx))
    noise_hat = grid.fft2(noise)
    kmag = np.sqrt(grid.K2)
    envelope = np.exp(-((kmag / sigma) ** 2))
    field = grid.ifft2_real(noise_hat * envelope)
    return field - field.mean()


def test_energy_budget_matches_dissipation_rate():
    """For unforced periodic 2D NS, dE/dt = -2*nu*enstrophy exactly (advection
    conserves energy; only viscosity dissipates it). Check this numerically
    on a broadband initial condition (not Taylor-Green, whose nonlinear term
    is trivially zero) via a central-difference estimate of dE/dt from saved
    snapshots against -2*nu*enstrophy(t)."""
    N = 48
    nu = 0.02
    grid = SpectralGrid(N, N)
    omega0 = _lowpass_random_field(grid, seed=1)

    p = SimParams(Nx=N, Ny=N, nu=nu, dt=2e-3, t_end=0.3, save_every=5, dealias=True)
    res = run(p, omega0)

    E = np.array([kinetic_energy(w, grid) for w in res.omega])
    Omega = np.array([enstrophy(w, grid) for w in res.omega])
    dE_dt_numeric = np.gradient(E, res.t)
    dE_dt_theory = -2.0 * nu * Omega

    rel_mismatch = np.abs(dE_dt_numeric - dE_dt_theory) / (np.abs(dE_dt_theory) + 1e-12)
    # exclude the two points at each end where the one-sided gradient stencil
    # is less accurate
    assert np.max(rel_mismatch[2:-2]) < 1e-3


def test_inviscid_energy_is_nearly_conserved_over_short_time():
    """With nu = 0 and no forcing, energy should be conserved by the
    continuous PDE; the dealiased pseudo-spectral discretisation with a
    small enough time step should conserve it to a small, bounded drift
    over a short integration."""
    N = 48
    grid = SpectralGrid(N, N)
    omega0 = _lowpass_random_field(grid, seed=2)

    p = SimParams(Nx=N, Ny=N, nu=0.0, dt=1e-3, t_end=0.1, save_every=10, dealias=True)
    res = run(p, omega0)
    E = np.array([kinetic_energy(w, grid) for w in res.omega])

    rel_drift = np.abs(E - E[0]) / E[0]
    assert np.max(rel_drift) < 1e-3


def test_enstrophy_is_nonnegative():
    N = 32
    grid = SpectralGrid(N, N)
    omega0 = _lowpass_random_field(grid, seed=3)
    assert enstrophy(omega0, grid) >= 0.0


def test_velocity_stays_divergence_free_through_a_run():
    N = 32
    grid = SpectralGrid(N, N)
    omega0 = _lowpass_random_field(grid, seed=4)
    p = SimParams(Nx=N, Ny=N, nu=0.01, dt=1e-3, t_end=0.02, save_every=1, dealias=True)
    res = run(p, omega0)
    for w in res.omega:
        assert max_divergence(w, grid) < 1e-9
