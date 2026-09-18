"""Spectral operators: streamfunction-Poisson solve, velocity recovery, dealiasing.

Governing relations (vorticity-streamfunction formulation, doubly periodic
domain):

    -nabla^2 psi = omega            (streamfunction Poisson equation)
    u = d(psi)/dy,   v = -d(psi)/dx  (velocity recovery; div(u) = 0 identically)

In Fourier space, with wavenumbers (kx, ky) and k^2 = kx^2 + ky^2:

    psi_hat = omega_hat / k^2,   psi_hat(0, 0) := 0
    u_hat = i*ky*psi_hat,   v_hat = -i*kx*psi_hat

The k = 0 mode of psi is a free constant (psi is only defined up to an
additive constant by the Poisson equation) and is set to zero by convention;
it does not affect the velocity field, which only involves derivatives of
psi.
"""
from __future__ import annotations

import numpy as np

from .grid import SpectralGrid


def poisson_solve(omega_hat: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    """Solve -nabla^2 psi = omega for psi_hat given omega_hat."""
    psi_hat = omega_hat * grid.K2_inv
    psi_hat[0, 0] = 0.0
    return psi_hat


def velocity_from_psi(psi_hat: np.ndarray, grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_hat, v_hat) = (d(psi)/dy, -d(psi)/dx) in Fourier space."""
    u_hat = 1j * grid.KY * psi_hat * grid.nyquist_mask
    v_hat = -1j * grid.KX * psi_hat * grid.nyquist_mask
    return u_hat, v_hat


def dealias_mask(grid: SpectralGrid) -> np.ndarray:
    """Boolean mask for the standard 2/3-rule dealiasing filter.

    Quadratic nonlinearities (u * omega_x, v * omega_y here) alias energy
    from wavenumber pairs (k1, k2) with |k1 + k2| beyond the resolvable
    range back into the resolved range. Truncating each spatial direction
    to the lowest 2/3 of its wavenumbers before forming the product removes
    all aliased contributions exactly, for a single quadratic nonlinearity
    (Orszag 1971).
    """
    kx_max = np.max(np.abs(grid.KX))
    ky_max = np.max(np.abs(grid.KY))
    cutoff_x = (2.0 / 3.0) * kx_max
    cutoff_y = (2.0 / 3.0) * ky_max
    return (np.abs(grid.KX) <= cutoff_x) & (np.abs(grid.KY) <= cutoff_y)
