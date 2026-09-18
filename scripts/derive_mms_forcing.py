"""Symbolic derivation of the manufactured-solution forcing term for the
vorticity-streamfunction Navier-Stokes solver.

A streamfunction built from a SINGLE Fourier mode gives an identically zero
nonlinear (Jacobian) term, because omega is then proportional to psi and
J(psi, omega) = const * J(psi, psi) = 0. That degenerate case (it is in
fact exactly the Taylor-Green vortex used elsewhere in this repo) cannot
exercise the advection term. This script derives the forcing required to
make a TWO-mode streamfunction an exact solution of the forced vorticity
equation, so the manufactured-solution test in tests/test_mms.py genuinely
checks the nonlinear term.

Run with: python scripts/derive_mms_forcing.py
Output is copied by hand into src/manufactured.py (kept as a plain closed
form there so the package has no runtime sympy dependency).
"""
from __future__ import annotations

import sympy as sp

x, y, t, nu = sp.symbols("x y t nu", real=True)
k1x, k1y, k2x, k2y = sp.symbols("k1x k1y k2x k2y", positive=True)
A1, A2, lam1, lam2 = sp.symbols("A1 A2 lam1 lam2", positive=True)

psi = (
    A1 * sp.cos(k1x * x) * sp.cos(k1y * y) * sp.exp(-lam1 * t)
    + A2 * sp.cos(k2x * x) * sp.sin(k2y * y) * sp.exp(-lam2 * t)
)

omega = -(sp.diff(psi, x, 2) + sp.diff(psi, y, 2))
omega = sp.simplify(omega)

u = sp.diff(psi, y)
v = -sp.diff(psi, x)

omega_t = sp.diff(omega, t)
omega_x = sp.diff(omega, x)
omega_y = sp.diff(omega, y)
laplacian_omega = sp.diff(omega, x, 2) + sp.diff(omega, y, 2)

advection = u * omega_x + v * omega_y
forcing = sp.simplify(omega_t + advection - nu * laplacian_omega)

print("omega =")
sp.pprint(omega)
print()
print("advection term u.grad(omega) =")
sp.pprint(sp.simplify(advection))
print()
print("forcing f = d(omega)/dt + u.grad(omega) - nu*laplacian(omega) =")
sp.pprint(forcing)
print()
print("forcing (expanded/collected) =")
sp.pprint(sp.trigsimp(sp.expand_trig(sp.expand(forcing))))
