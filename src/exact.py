"""Exact (analytical) 2D Taylor-Green vortex solution.

On the periodic domain [0, 2*pi) x [0, 2*pi), for integer wavenumber k,

    u(x,y,t) =  cos(k*x)*sin(k*y)*exp(-2*nu*k^2*t)
    v(x,y,t) = -sin(k*x)*cos(k*y)*exp(-2*nu*k^2*t)

is an EXACT solution of the 2D incompressible Navier-Stokes equations. Its
vorticity is

    omega(x,y,t) = -2*k*cos(k*x)*cos(k*y)*exp(-2*nu*k^2*t)

A special (and easily verified) property of this flow is that the
nonlinear term vanishes identically: u.grad(omega) = 0 for all x, y, t,
because omega is proportional to a single Laplacian eigenfunction of the
streamfunction, psi = -(1/k)*cos(k*x)*cos(k*y)*exp(-2*nu*k^2*t) (the
Jacobian J(psi, c*psi) = c*J(psi, psi) = 0 for any constant c). Vorticity
therefore obeys pure linear diffusion, omega_t = nu*nabla^2(omega), which
is exactly what the integrating-factor part of the time stepper solves
with no time-discretisation error. This makes the Taylor-Green vortex an
excellent check of the spectral derivatives, the streamfunction-Poisson
solve, and the diffusion/time-integration machinery, but it does NOT
exercise the nonlinear advection term; see manufactured.py and
tests/test_mms.py for a check that does.
"""
from __future__ import annotations

import numpy as np


def taylor_green_omega(X: np.ndarray, Y: np.ndarray, t: float, nu: float, k: int = 1) -> np.ndarray:
    decay = np.exp(-2.0 * nu * k**2 * t)
    return -2.0 * k * np.cos(k * X) * np.cos(k * Y) * decay


def taylor_green_velocity(
    X: np.ndarray, Y: np.ndarray, t: float, nu: float, k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    decay = np.exp(-2.0 * nu * k**2 * t)
    u = np.cos(k * X) * np.sin(k * Y) * decay
    v = -np.sin(k * X) * np.cos(k * Y) * decay
    return u, v
