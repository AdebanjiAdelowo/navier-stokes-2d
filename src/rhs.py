"""Nonlinear advection term (and optional forcing) of the vorticity equation.

The full vorticity-transport equation is

    d(omega)/dt + u.grad(omega) = nu * nabla^2(omega) + f

This module computes only the nonlinear advection term (with a sign flip,
i.e. -u.grad(omega)) plus any forcing; the linear viscous term is handled
separately and exactly by the integrating-factor time stepper in
``solver.py``.

The advection term is evaluated pseudo-spectrally: u, v, omega_x, omega_y
are computed in Fourier space, transformed to physical space by inverse
FFT, multiplied pointwise, and transformed back. The quadratic product
u*omega_x + v*omega_y aliases energy from unresolved to resolved
wavenumbers unless dealiased; the standard 2/3-rule mask (Orszag 1971)
removes this exactly for a single quadratic nonlinearity.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .grid import SpectralGrid
from .operators import poisson_solve, velocity_from_psi


def vorticity_rhs(
    omega_hat: np.ndarray,
    grid: SpectralGrid,
    mask: Optional[np.ndarray] = None,
    forcing_hat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return N_hat = -(u.grad(omega))_hat [+ forcing_hat], dealiased if mask given."""
    psi_hat = poisson_solve(omega_hat, grid)
    u_hat, v_hat = velocity_from_psi(psi_hat, grid)

    omega_x_hat = 1j * grid.KX * omega_hat * grid.nyquist_mask
    omega_y_hat = 1j * grid.KY * omega_hat * grid.nyquist_mask

    u = grid.ifft2_real(u_hat)
    v = grid.ifft2_real(v_hat)
    omega_x = grid.ifft2_real(omega_x_hat)
    omega_y = grid.ifft2_real(omega_y_hat)

    advection = u * omega_x + v * omega_y
    advection_hat = grid.fft2(advection)
    if mask is not None:
        advection_hat = advection_hat * mask

    n_hat = -advection_hat
    if forcing_hat is not None:
        n_hat = n_hat + forcing_hat
    return n_hat


ForcingFn = Callable[[float], np.ndarray]
