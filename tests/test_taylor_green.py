"""Taylor-Green vortex: exact solution of 2D Navier-Stokes.

Verifies the linear machinery (spectral derivatives, streamfunction-Poisson
solve, exact integrating-factor diffusion). Because u.grad(omega) = 0
identically for this flow (see src/exact.py docstring), this benchmark does
NOT exercise the nonlinear advection term; see test_mms.py for that.
"""
import numpy as np

from src import SimParams, SpectralGrid, run, taylor_green_omega, max_divergence


def test_taylor_green_matches_exact_solution():
    N = 32
    nu = 0.05
    p = SimParams(Nx=N, Ny=N, nu=nu, dt=1e-3, t_end=0.05, save_every=10**9)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.0, nu, k=1)

    res = run(p, omega0)
    exact_final = taylor_green_omega(grid.X, grid.Y, res.t[-1], nu, k=1)
    rel_err = np.linalg.norm(res.omega[-1] - exact_final) / np.linalg.norm(exact_final)

    # Nonlinear term is exactly zero for TGV, and diffusion is integrated
    # exactly by the integrating factor, so the only error source is
    # floating-point roundoff: expect it at the machine-epsilon level.
    assert rel_err < 1e-10


def test_taylor_green_velocity_field_is_divergence_free():
    N = 32
    nu = 0.05
    p = SimParams(Nx=N, Ny=N, nu=nu, dt=1e-3, t_end=0.02, save_every=10**9)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.0, nu, k=1)
    res = run(p, omega0)
    assert max_divergence(res.omega[-1], grid) < 1e-10


def test_taylor_green_nonlinear_term_is_zero():
    """Direct check of the property that makes TGV special: u.grad(omega) = 0."""
    from src.rhs import vorticity_rhs

    N = 32
    p = SimParams(Nx=N, Ny=N)
    grid = SpectralGrid(N, N, p.Lx, p.Ly)
    omega0 = taylor_green_omega(grid.X, grid.Y, 0.3, nu=0.05, k=1)
    omega_hat = grid.fft2(omega0)
    n_hat = vorticity_rhs(omega_hat, grid, mask=None, forcing_hat=None)
    n_physical = grid.ifft2_real(n_hat)
    assert np.max(np.abs(n_physical)) < 1e-9
