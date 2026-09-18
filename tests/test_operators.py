import numpy as np
import pytest

from src import SpectralGrid, poisson_solve, velocity_from_psi, dealias_mask


def test_poisson_solve_manufactured_single_mode():
    """psi = cos(kx*x)*cos(ky*y) => omega = -laplacian(psi) = (kx^2+ky^2)*psi.
    Solving poisson_solve(omega_hat) should recover psi_hat exactly (to fp precision)."""
    N = 32
    grid = SpectralGrid(N, N)
    kx, ky = 3, 2
    psi_exact = np.cos(kx * grid.X) * np.cos(ky * grid.Y)
    omega = (kx**2 + ky**2) * psi_exact

    omega_hat = grid.fft2(omega)
    psi_hat = poisson_solve(omega_hat, grid)
    psi_numeric = grid.ifft2_real(psi_hat)

    assert np.allclose(psi_numeric, psi_exact, atol=1e-10)


def test_poisson_solve_zero_mode_is_zero():
    N = 16
    grid = SpectralGrid(N, N)
    omega = np.ones((N, N)) * 5.0  # pure zero-wavenumber field
    omega_hat = grid.fft2(omega)
    psi_hat = poisson_solve(omega_hat, grid)
    assert psi_hat[0, 0] == 0.0


def test_velocity_from_psi_matches_analytical_derivatives():
    N = 32
    grid = SpectralGrid(N, N)
    kx, ky = 2, 3
    psi = np.cos(kx * grid.X) * np.cos(ky * grid.Y)
    psi_hat = grid.fft2(psi)
    u_hat, v_hat = velocity_from_psi(psi_hat, grid)
    u = grid.ifft2_real(u_hat)
    v = grid.ifft2_real(v_hat)

    u_exact = -ky * np.cos(kx * grid.X) * np.sin(ky * grid.Y)
    v_exact = kx * np.sin(kx * grid.X) * np.cos(ky * grid.Y)

    assert np.allclose(u, u_exact, atol=1e-10)
    assert np.allclose(v, v_exact, atol=1e-10)


def test_velocity_from_psi_is_divergence_free():
    """u = psi_y, v = -psi_x implies div(u) = 0 identically in spectral space."""
    N = 24
    grid = SpectralGrid(N, N)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((N, N))
    psi_hat = grid.fft2(psi)
    u_hat, v_hat = velocity_from_psi(psi_hat, grid)
    div_hat = 1j * grid.KX * u_hat + 1j * grid.KY * v_hat
    div = grid.ifft2_real(div_hat)
    assert np.max(np.abs(div)) < 1e-10


def test_dealias_mask_shape_and_symmetry():
    N = 16
    grid = SpectralGrid(N, N)
    mask = dealias_mask(grid)
    assert mask.shape == (N, N)
    assert mask.dtype == bool
    # zero wavenumber always kept
    assert mask[0, 0]
    # roughly one third of modes in each direction removed by the 2/3 rule
    kept_fraction = mask.mean()
    assert 0.3 < kept_fraction < 0.6


def test_grid_nyquist_mask_zeroes_nyquist_row_and_column():
    N = 16  # even -> has a true Nyquist frequency
    grid = SpectralGrid(N, N)
    assert np.all(grid.nyquist_mask[:, N // 2] == 0.0)
    assert np.all(grid.nyquist_mask[N // 2, :] == 0.0)
