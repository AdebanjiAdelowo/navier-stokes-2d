"""Time integration: integrating-factor RK4 for the vorticity equation.

    d(omega_hat)/dt = L * omega_hat + N(omega_hat, t)

with L = -nu * k^2 (diagonal, from the viscous term) treated exactly via an
integrating factor, and N (the nonlinear advection term, plus any forcing)
stepped with classical explicit RK4. This is the standard "integrating
factor RK4" (IFRK4) scheme used for stiff-diffusion / non-stiff-advection
spectral PDEs (see e.g. Trefethen, "Spectral Methods in MATLAB", 2000, and
Canuto et al., "Spectral Methods: Fundamentals in Single Domains", 2006).
Because the viscous term is integrated exactly, the time-step restriction
comes only from the explicit treatment of advection (CFL-like), not from
diffusion, however fine the grid.

Derivation (writing y = omega_hat, h = dt, E_half = exp(L h/2),
E_full = exp(L h)):

    N1 = N(y_n, t_n)
    Y2 = E_half*y_n + (h/2)*E_half*N1;              N2 = N(Y2, t_n + h/2)
    Y3 = E_half*y_n + (h/2)*N2;                      N3 = N(Y3, t_n + h/2)
    Y4 = E_full*y_n + h*E_half*N3;                   N4 = N(Y4, t_n + h)
    y_{n+1} = E_full*y_n + (h/6)*(E_full*N1 + 2*E_half*N2 + 2*E_half*N3 + N4)

which reduces to classical RK4 when L = 0 (E_half = E_full = 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .grid import SpectralGrid
from .operators import dealias_mask
from .params import SimParams
from .rhs import vorticity_rhs


@dataclass
class RunResult:
    t: np.ndarray
    omega: np.ndarray  # shape (n_saved, Ny, Nx), physical space
    grid: SpectralGrid = field(repr=False)


def run(
    params: SimParams,
    omega0: np.ndarray,
    forcing: Optional[Callable[[float], np.ndarray]] = None,
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
) -> RunResult:
    """Integrate the vorticity equation from omega0 for t in [0, t_end].

    omega0: initial vorticity, physical space, shape (Ny, Nx).
    forcing: optional callable t -> omega_hat-space forcing array (Ny, Nx complex).
    callback: optional callable(step, t, omega_hat) invoked after every accepted step,
        for diagnostics that need to run online rather than from saved snapshots.
    """
    grid = SpectralGrid(params.Nx, params.Ny, params.Lx, params.Ly)
    mask = dealias_mask(grid) if params.dealias else None

    L = -params.nu * grid.K2
    h = params.dt
    E_half = np.exp(L * h / 2.0)
    E_full = np.exp(L * h)

    n_steps = int(round(params.t_end / params.dt))
    omega_hat = grid.fft2(omega0)

    t_list = [0.0]
    omega_list = [omega0.copy()]
    t = 0.0

    for step in range(1, n_steps + 1):
        f_n = forcing(t) if forcing is not None else None
        f_mid = forcing(t + h / 2.0) if forcing is not None else None
        f_next = forcing(t + h) if forcing is not None else None

        N1 = vorticity_rhs(omega_hat, grid, mask, f_n)
        Y2 = E_half * omega_hat + (h / 2.0) * E_half * N1
        N2 = vorticity_rhs(Y2, grid, mask, f_mid)
        Y3 = E_half * omega_hat + (h / 2.0) * N2
        N3 = vorticity_rhs(Y3, grid, mask, f_mid)
        Y4 = E_full * omega_hat + h * E_half * N3
        N4 = vorticity_rhs(Y4, grid, mask, f_next)

        omega_hat = E_full * omega_hat + (h / 6.0) * (
            E_full * N1 + 2.0 * E_half * N2 + 2.0 * E_half * N3 + N4
        )
        t = step * h

        if callback is not None:
            callback(step, t, omega_hat)

        if step % params.save_every == 0 or step == n_steps:
            t_list.append(t)
            omega_list.append(grid.ifft2_real(omega_hat))

    return RunResult(t=np.array(t_list), omega=np.array(omega_list), grid=grid)
