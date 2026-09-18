"""Physical diagnostics computed from a vorticity field: energy, enstrophy, divergence.

Kinetic energy:  E(t) = (1/2) integral (u^2 + v^2) dA
Enstrophy:       Omega(t) = (1/2) integral omega^2 dA
Energy budget (no forcing):  dE/dt = -2*nu*Omega   (exact identity for 2D NS
    with periodic BCs; advection conserves energy, only viscosity dissipates it)

Integrals are approximated on the uniform periodic grid by the midpoint /
trapezoidal rule (equivalent here since the grid is periodic), which is
spectrally accurate for smooth periodic fields.
"""
from __future__ import annotations

import numpy as np

from .grid import SpectralGrid
from .operators import poisson_solve, velocity_from_psi


def _velocity(omega: np.ndarray, grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray]:
    omega_hat = grid.fft2(omega)
    psi_hat = poisson_solve(omega_hat, grid)
    u_hat, v_hat = velocity_from_psi(psi_hat, grid)
    return grid.ifft2_real(u_hat), grid.ifft2_real(v_hat)


def kinetic_energy(omega: np.ndarray, grid: SpectralGrid) -> float:
    u, v = _velocity(omega, grid)
    cell_area = grid.dx * grid.dy
    return 0.5 * float(np.sum(u**2 + v**2)) * cell_area


def enstrophy(omega: np.ndarray, grid: SpectralGrid) -> float:
    cell_area = grid.dx * grid.dy
    return 0.5 * float(np.sum(omega**2)) * cell_area


def divergence_field(omega: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    omega_hat = grid.fft2(omega)
    psi_hat = poisson_solve(omega_hat, grid)
    u_hat, v_hat = velocity_from_psi(psi_hat, grid)
    div_hat = 1j * grid.KX * u_hat * grid.nyquist_mask + 1j * grid.KY * v_hat * grid.nyquist_mask
    return grid.ifft2_real(div_hat)


def divergence_l2(omega: np.ndarray, grid: SpectralGrid) -> float:
    div = divergence_field(omega, grid)
    cell_area = grid.dx * grid.dy
    return float(np.sqrt(np.sum(div**2) * cell_area))


def max_divergence(omega: np.ndarray, grid: SpectralGrid) -> float:
    return float(np.max(np.abs(divergence_field(omega, grid))))
