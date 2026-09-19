"""Galerkin projection of the vorticity-streamfunction 2D Navier-Stokes equations onto a POD basis,
and a matrix-exponential integrating-factor RK4 integrator for the resulting reduced ODE system.

## No mean-centering

This project's training data is a family of FREELY DECAYING (unforced) trajectories: the flow relaxes
monotonically toward the zero state, so there is no statistically steady "mean flow" that a
time-averaged base state would meaningfully represent (unlike, e.g., POD-Galerkin ROMs of a
statistically stationary forced/sustained turbulent flow, where the time-mean is itself a physically
meaningful base state). Mean-subtraction is therefore NOT used here: the ansatz is the pure POD-mode
expansion

    omega_r(x, t) = sum_{i=1}^r a_i(t) phi_i(x),

not the affine omega_r = omega_bar + sum a_i phi_i form. This removes the constant-in-a and
linear-in-a "mean-coupling" terms from the reduced equations below, leaving a purely quadratic
(homogeneous) reduced dynamical system, without loss of generality for this flow class.

## Reduced-equation derivation

Substituting the ansatz into the vorticity equation d(omega)/dt + u.grad(omega) = nu*laplacian(omega)
and taking the L2 inner product (`pod.l2_inner_product`) with each test mode phi_k, using
L2-orthonormality of the POD basis:

    da_k/dt = <nu*laplacian(omega_r) - u(omega_r).grad(omega_r), phi_k>

Both u(omega_r) and grad(omega_r) are LINEAR in the coefficient vector a (the streamfunction Poisson
solve and spectral differentiation are linear operators), so:

    u(omega_r) = sum_i a_i * u_i,      grad(omega_r) = sum_j a_j * grad(phi_i)

with u_i = velocity recovered from mode phi_i alone (i.e. solve -laplacian(psi_i) = phi_i, then
u_i = d(psi_i)/dy, v_i = -d(psi_i)/dx -- exactly the FOM's own `poisson_solve` / `velocity_from_psi`
applied to a mode field instead of a full vorticity field). Substituting and collecting terms:

- **Linear viscous operator**: L_kj = <laplacian(phi_j), phi_k>  (r x r matrix, precomputed once).
- **Quadratic convective tensor**: Q_kij = -<u_i * d(phi_j)/dx + v_i * d(phi_j)/dy, phi_k>
  (r x r x r tensor, precomputed once; the leading minus sign folds in the vorticity equation's
  own -u.grad(omega) sign).
- **No constant or forcing term** (no mean-centering, no external forcing in this flow).

giving the closed reduced dynamical system

    da_k/dt = nu * sum_j L_kj a_j + sum_{i,j} Q_kij a_i a_j,   k = 1..r.

If dealiasing is used in the FOM (the 2/3-rule mask, `..operators.dealias_mask`), the SAME mask is
applied when forming each precomputed advection-product field before projecting it onto phi_k, so
that (see `verify_operators`) evaluating these tensors reproduces EXACTLY what projecting the FOM's
own dealiased nonlinear evaluation would give -- this is an exact algebraic identity (FFT, masking and
the L2 inner product are all linear operations, and the physical-space product of the affine-in-a
fields u(omega_r), grad(omega_r) distributes exactly termwise), not an approximation, which is why
`verify_operators` below checks agreement to numerical (not truncation-level) precision.

## Integrator

The same integrating-factor RK4 (IFRK4) scheme used by the FOM (`..solver.run`), generalized from a
scalar diagonal exponential to a dense matrix exponential exp(nu*L*h) (computed once via
`scipy.linalg.expm`, since r is small), treating the quadratic term explicitly with RK4 exactly as the
FOM treats advection explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..grid import SpectralGrid
from ..operators import dealias_mask, poisson_solve, velocity_from_psi
from ..rhs import vorticity_rhs
from .pod import cell_area, l2_inner_product


def spectral_partials(f: np.ndarray, grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray]:
    """(df/dx, df/dy) via spectral differentiation, matching src/rhs.py's internal treatment of
    omega_x, omega_y exactly (same FFT, same 1j*K, same Nyquist masking)."""
    f_hat = grid.fft2(f)
    fx = grid.ifft2_real(1j * grid.KX * f_hat * grid.nyquist_mask)
    fy = grid.ifft2_real(1j * grid.KY * f_hat * grid.nyquist_mask)
    return fx, fy


def velocity_of(f: np.ndarray, grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray]:
    """Velocity recovered from a scalar field via the streamfunction-Poisson relation, matching
    src/rhs.py's treatment of the vorticity field exactly (reuses the FOM's own operators)."""
    f_hat = grid.fft2(f)
    psi_hat = poisson_solve(f_hat, grid)
    u_hat, v_hat = velocity_from_psi(psi_hat, grid)
    return grid.ifft2_real(u_hat), grid.ifft2_real(v_hat)


def laplacian_of(f: np.ndarray, grid: SpectralGrid) -> np.ndarray:
    f_hat = grid.fft2(f)
    return grid.ifft2_real(-grid.K2 * f_hat)


@dataclass
class GalerkinOperators:
    A_lin: np.ndarray  # (r, r) = nu * L
    Q: np.ndarray  # (r, r, r)
    r: int
    nu: float
    dealias: bool


def build_galerkin_operators(Phi: np.ndarray, grid: SpectralGrid, nu: float,
                              dealias: bool = True) -> GalerkinOperators:
    """Precompute the reduced linear viscous operator and quadratic convective tensor from the POD
    modes (see module docstring for the derivation). Phi: (N_dof, r) L2-orthonormal POD modes."""
    r = Phi.shape[1]
    ny, nx = grid.Ny, grid.Nx
    mask = dealias_mask(grid) if dealias else None
    c = cell_area(grid)

    modes = [Phi[:, i].reshape(ny, nx) for i in range(r)]
    mode_u = np.empty((r, ny, nx))
    mode_v = np.empty((r, ny, nx))
    mode_fx = np.empty((r, ny, nx))
    mode_fy = np.empty((r, ny, nx))
    for i in range(r):
        mode_u[i], mode_v[i] = velocity_of(modes[i], grid)
        mode_fx[i], mode_fy[i] = spectral_partials(modes[i], grid)

    L = np.zeros((r, r))
    for j in range(r):
        lap_j = laplacian_of(modes[j], grid)
        for k in range(r):
            L[k, j] = l2_inner_product(lap_j, modes[k], grid)

    Q = np.zeros((r, r, r))
    for i in range(r):
        for j in range(r):
            adv_field = mode_u[i] * mode_fx[j] + mode_v[i] * mode_fy[j]
            if mask is not None:
                adv_field = grid.ifft2_real(grid.fft2(adv_field) * mask)
            for k in range(r):
                Q[k, i, j] = -c * float(np.sum(adv_field * modes[k]))

    return GalerkinOperators(A_lin=nu * L, Q=Q, r=r, nu=nu, dealias=dealias)


def reduced_rhs(a: np.ndarray, ops: GalerkinOperators) -> np.ndarray:
    """da/dt = A_lin @ a + Q : a x a  (the precomputed-tensor evaluation, O(r^3))."""
    return ops.A_lin @ a + np.einsum("kij,i,j->k", ops.Q, a, a)


def fom_rhs_field(omega: np.ndarray, grid: SpectralGrid, nu: float,
                   mask: np.ndarray | None) -> np.ndarray:
    """Full right-hand side d(omega)/dt = nu*laplacian(omega) - u.grad(omega), evaluated by the FOM's
    own operators (`vorticity_rhs` for advection, direct spectral Laplacian for viscosity) -- used
    both by `verify_operators` and by the "reconstruct + call FOM RHS" baseline ROM."""
    omega_hat = grid.fft2(omega)
    adv_hat = vorticity_rhs(omega_hat, grid, mask)
    lap_hat = -grid.K2 * omega_hat
    return grid.ifft2_real(nu * lap_hat + adv_hat)


def projected_fom_rhs(a: np.ndarray, Phi: np.ndarray, grid: SpectralGrid, nu: float,
                       dealias: bool) -> np.ndarray:
    """Reconstruct omega_r = Phi @ a, evaluate the FULL FOM right-hand side at that field, and
    project the result back onto the POD basis. Used only for independent verification of the
    precomputed tensors (`verify_operators`) and for the naive reconstruct-and-call-FOM-RHS baseline
    ROM -- NOT part of the fast tensorized ROM's online evaluation."""
    ny, nx = grid.Ny, grid.Nx
    mask = dealias_mask(grid) if dealias else None
    omega = (Phi @ a).reshape(ny, nx)
    rhs_field = fom_rhs_field(omega, grid, nu, mask)
    c = cell_area(grid)
    return c * (Phi.T @ rhs_field.reshape(-1))


def verify_operators(Phi: np.ndarray, ops: GalerkinOperators, grid: SpectralGrid,
                      n_trials: int = 5, seed: int = 0) -> np.ndarray:
    """For random reduced states a, compare the precomputed-tensor reduced RHS against the projected
    FOM RHS evaluated at the reconstructed field. Returns the array of max-absolute discrepancies
    (one per trial); these should be at numerical-precision (machine-epsilon-scale relative to the
    magnitude of the RHS terms), not merely "small" -- this is an exact algebraic identity (see module
    docstring), so any larger discrepancy indicates a genuine implementation bug, not expected
    truncation error."""
    rng = np.random.default_rng(seed)
    discrepancies = []
    for _ in range(n_trials):
        a = rng.uniform(-1.0, 1.0, ops.r)
        rhs_tensor = reduced_rhs(a, ops)
        rhs_projected = projected_fom_rhs(a, Phi, grid, ops.nu, ops.dealias)
        discrepancies.append(float(np.max(np.abs(rhs_tensor - rhs_projected))))
    return np.array(discrepancies)


@dataclass
class ROMResult:
    t: np.ndarray  # (n_saved,)
    a: np.ndarray  # (n_saved, r)
    stable: bool  # False if the trajectory was aborted due to non-finite coefficients


def integrate_rom(ops: GalerkinOperators, a0: np.ndarray, dt: float, t_end: float,
                   save_every: int = 1) -> ROMResult:
    """Matrix-exponential integrating-factor RK4 for da/dt = A_lin @ a + Q:a x a (see module
    docstring). Aborts early (flagging `stable=False`) if coefficients become non-finite, rather than
    letting NaNs silently propagate through downstream diagnostics."""
    E_half = expm(ops.A_lin * dt / 2.0)
    E_full = expm(ops.A_lin * dt)

    def N(a):
        return np.einsum("kij,i,j->k", ops.Q, a, a)

    n_steps = int(round(t_end / dt))
    a = a0.copy()
    t_list = [0.0]
    a_list = [a0.copy()]
    stable = True

    for step in range(1, n_steps + 1):
        N1 = N(a)
        Y2 = E_half @ a + (dt / 2.0) * (E_half @ N1)
        N2 = N(Y2)
        Y3 = E_half @ a + (dt / 2.0) * N2
        N3 = N(Y3)
        Y4 = E_full @ a + dt * (E_half @ N3)
        N4 = N(Y4)
        a = E_full @ a + (dt / 6.0) * (E_full @ N1 + 2.0 * (E_half @ N2) + 2.0 * (E_half @ N3) + N4)

        if not np.all(np.isfinite(a)):
            stable = False
            break

        t = step * dt
        if step % save_every == 0 or step == n_steps:
            t_list.append(t)
            a_list.append(a.copy())

    return ROMResult(t=np.array(t_list), a=np.array(a_list), stable=stable)
