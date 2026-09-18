"""Reproducible initial conditions for the decaying-2D-turbulence benchmark.

There is no closed-form solution for freely decaying 2D turbulence, so this
initial condition is not an exact-solution benchmark like the Taylor-Green
vortex; it is used for physical-diagnostic validation (energy/enstrophy
decay, the energy budget dE/dt = -2*nu*enstrophy) and for a self-convergence
study against a high-resolution reference (see scripts/convergence_spatial.py).

The field is an isotropic random vorticity realisation with Fourier
amplitude shaped by a bump function peaked at wavenumber k0,

    A(k) = (k/k0)^2 * exp(1 - (k/k0)^2)

which vanishes at k = 0 and decays at high k, concentrating variance at
intermediate scales -- the standard qualitative setup for freely decaying
2D turbulence (energy initially concentrated away from both the largest and
smallest resolved scales; cf. the general practice described in
McWilliams, J. C. (1984), "The emergence of isolated coherent vortices in
turbulent flow", J. Fluid Mech. 146, 21-43, though the specific envelope
used here is our own simple choice, not a reproduction of that paper's
exact spectrum). Starting from real-valued Gaussian white noise and
applying a real, isotropic filter in Fourier space keeps the field real
without needing to impose Hermitian symmetry by hand.
"""
from __future__ import annotations

import numpy as np

from .grid import SpectralGrid


def random_vorticity_field(
    grid: SpectralGrid,
    seed: int,
    k0: float = 6.0,
    target_enstrophy: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((grid.Ny, grid.Nx))
    noise_hat = grid.fft2(noise)

    kmag = np.sqrt(grid.K2)
    with np.errstate(divide="ignore", invalid="ignore"):
        envelope = (kmag / k0) ** 2 * np.exp(1.0 - (kmag / k0) ** 2)
    envelope[0, 0] = 0.0  # zero-mean field: no k=0 vorticity component

    omega_hat = noise_hat * envelope
    omega = grid.ifft2_real(omega_hat)
    omega -= omega.mean()

    current_enstrophy = 0.5 * np.mean(omega**2) * grid.Lx * grid.Ly
    scale = np.sqrt(target_enstrophy / current_enstrophy)
    return omega * scale
