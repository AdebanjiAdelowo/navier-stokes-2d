"""General solver sanity checks: reproducibility, shapes, forcing plumbing."""
import numpy as np

from src import SimParams, SpectralGrid, run, taylor_green_omega


def test_run_is_deterministic():
    N = 24
    p = SimParams(Nx=N, Ny=N, nu=0.05, dt=1e-3, t_end=0.02, save_every=2)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.0, p.nu, k=1)

    res1 = run(p, omega0)
    res2 = run(p, omega0)
    assert np.array_equal(res1.omega, res2.omega)


def test_output_shapes_and_save_every():
    N = 16
    save_every = 4
    n_steps = 20
    dt = 1e-3
    p = SimParams(Nx=N, Ny=N, nu=0.01, dt=dt, t_end=n_steps * dt, save_every=save_every)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.0, p.nu, k=1)

    res = run(p, omega0)
    expected_saves = n_steps // save_every  # plus t=0
    assert res.omega.shape[0] == expected_saves + 1
    assert res.omega.shape[1:] == (N, N)
    assert res.t[0] == 0.0
    assert abs(res.t[-1] - p.t_end) < 1e-12


def test_zero_forcing_matches_no_forcing():
    N = 16
    p = SimParams(Nx=N, Ny=N, nu=0.02, dt=1e-3, t_end=0.02, save_every=10**9)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.0, p.nu, k=1)

    res_no_forcing = run(p, omega0)
    res_zero_forcing = run(p, omega0, forcing=lambda t: np.zeros((N, N), dtype=complex))
    assert np.allclose(res_no_forcing.omega, res_zero_forcing.omega)


def test_callback_is_invoked_every_step():
    N = 16
    n_steps = 7
    dt = 1e-3
    p = SimParams(Nx=N, Ny=N, nu=0.01, dt=dt, t_end=n_steps * dt, save_every=100)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.0, p.nu, k=1)

    calls = []
    run(p, omega0, callback=lambda step, t, omega_hat: calls.append(step))
    assert calls == list(range(1, n_steps + 1))


def test_solution_remains_finite_and_bounded_for_smoke_config():
    N = 32
    grid = SpectralGrid(N, N)
    rng = np.random.default_rng(0)
    noise_hat = grid.fft2(rng.standard_normal((N, N)))
    envelope = np.exp(-(np.sqrt(grid.K2) / 6.0) ** 2)
    omega0 = grid.ifft2_real(noise_hat * envelope)

    p = SimParams(Nx=N, Ny=N, nu=0.02, dt=2e-3, t_end=0.1, save_every=10**9)
    res = run(p, omega0)
    assert np.all(np.isfinite(res.omega))
    assert np.max(np.abs(res.omega)) < 100 * np.max(np.abs(omega0))
