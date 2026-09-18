"""Two-mode manufactured solution for verifying the nonlinear advection term.

A streamfunction built from a single Fourier mode (e.g. the Taylor-Green
vortex in ``exact.py``) has omega proportional to psi, which makes the
Jacobian nonlinear term u.grad(omega) identically zero for any nu, at any
resolution. That is a genuine property of the Taylor-Green flow, not a
numerical accident, but it means TGV alone cannot verify the nonlinear
term is implemented correctly: a solver with a sign error or a missing
term in ``rhs.vorticity_rhs`` would still pass a TGV check.

This module manufactures a streamfunction that is the SUM of two Fourier
modes with different wavevectors, so omega is not proportional to psi and
the advection term is genuinely nonzero. The forcing required to make this
an exact solution of the forced vorticity equation

    d(omega)/dt + u.grad(omega) = nu*nabla^2(omega) + f

is derived symbolically with sympy (see scripts/derive_mms_forcing.py for
the standalone derivation script) and evaluated here via ``sympy.lambdify``
so there is no hand-transcribed algebra to get wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

_x, _y, _t, _nu = sp.symbols("x y t nu", real=True)


@dataclass
class ManufacturedSolution:
    """omega_exact(x,y,t) and forcing(x,y,t) for a two-Fourier-mode manufactured flow.

    psi = A1*cos(k1x*x)*cos(k1y*y)*exp(-lam1*t) + A2*cos(k2x*x)*sin(k2y*y)*exp(-lam2*t)
    """

    A1: float = 1.0
    A2: float = 0.7
    k1x: float = 1.0
    k1y: float = 1.0
    k2x: float = 2.0
    k2y: float = 1.0
    lam1: float = 0.3
    lam2: float = 0.5
    nu: float = 0.05

    def __post_init__(self) -> None:
        psi = (
            self.A1 * sp.cos(self.k1x * _x) * sp.cos(self.k1y * _y) * sp.exp(-self.lam1 * _t)
            + self.A2 * sp.cos(self.k2x * _x) * sp.sin(self.k2y * _y) * sp.exp(-self.lam2 * _t)
        )
        omega = -(sp.diff(psi, _x, 2) + sp.diff(psi, _y, 2))
        u = sp.diff(psi, _y)
        v = -sp.diff(psi, _x)
        advection = u * sp.diff(omega, _x) + v * sp.diff(omega, _y)
        laplacian_omega = sp.diff(omega, _x, 2) + sp.diff(omega, _y, 2)
        forcing = sp.diff(omega, _t) + advection - self.nu * laplacian_omega

        self._omega_fn = sp.lambdify((_x, _y, _t), omega, "numpy")
        self._forcing_fn = sp.lambdify((_x, _y, _t), sp.simplify(forcing), "numpy")

    def omega_exact(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(self._omega_fn(X, Y, t), dtype=float) * np.ones_like(X)

    def forcing(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(self._forcing_fn(X, Y, t), dtype=float) * np.ones_like(X)
