"""Discrete Empirical Interpolation Method (DEIM) hyper-reduction of the nonlinear advection term
(Chaturantabut & Sorensen, 2010, "Nonlinear model reduction via discrete empirical interpolation",
SIAM J. Sci. Comput. 32(5), 2737-2764).

## Motivation, and why this is a genuinely separate question from the state-POD rank

Section 6's tensorized Galerkin ROM already reduces the (exactly quadratic) advection term to a
PRECOMPUTED r x r x r tensor contraction -- an EXACT reformulation for this nonlinearity, with O(r^3)
online cost and NO grid-sized (O(N) or O(N log N)) operations at all. DEIM exists for the harder case
where the nonlinearity does NOT admit such an exact closed-form tensor (a general, non-polynomial
f(omega)): it approximates f(omega) by evaluating it at only m << N "cleverly chosen" grid points and
interpolating, rather than reducing it in closed form. Because the interpolation basis Psi and point
set are built from a SEPARATE SVD of nonlinear-term snapshots (not the state POD basis Phi), the DEIM
rank m is an independent hyperparameter from the state POD rank r -- this module and
scripts/rom_hyperreduction.py study it as such, never conflating the two (section 13/21).

## Construction

1. **Nonlinear snapshot matrix**: F_j = N(omega_j) = -(u(omega_j).grad(omega_j)) evaluated (dealiased,
   matching the FOM's own treatment) at each TRAINING snapshot omega_j -- a genuinely different
   snapshot ensemble from the state-vorticity one, built with `build_nonlinear_snapshots`.
2. **Nonlinear POD basis** Psi (N_dof x m): the leading m left singular vectors of the (raw, unit-
   weighted -- see note below) nonlinear snapshot matrix.
3. **Greedy interpolation-point selection** (Chaturantabut & Sorensen, Algorithm 1): points
   p_1, ..., p_m chosen one at a time to greedily minimize the interpolation error of Psi's own
   columns at each step.
4. **DEIM approximation**: N(omega) ~= Psi (P^T Psi)^{-1} P^T N(omega), i.e. N is evaluated ONLY at
   the m selected points and interpolated through the fixed matrix U = Psi (P^T Psi)^{-1}.

## Online reduced-nonlinear evaluation without any grid-sized operation

Projected onto the state POD basis, the DEIM-approximated nonlinear contribution to da/dt is
D @ N(omega)|_points, with the fixed (r x m) matrix D = Phi^T @ U precomputed offline. Evaluating
N(omega_r(a)) at the m DEIM points needs u, v, omega_x, omega_y ONLY at those m points; because these
are affine-in-a combinations of the (offline-precomputed, grid-wide) mode fields
u_i, v_i, phi_i_x, phi_i_y (`galerkin.py`), their values AT the m fixed points are themselves fixed
(r x m) arrays that can be indexed for free at every ROM step -- no FFT, no full-grid reconstruction,
matching the "genuinely reduces online cost" requirement of section 12. This is contrasted directly,
in scripts/rom_hyperreduction.py, against the explicit "naive" baseline that reconstructs the full
field and calls the FOM's own nonlinear evaluation at every step (`galerkin.projected_fom_rhs`),
where DEIM's benefit (if any) would actually show up.

Note on inner-product weighting: the nonlinear-snapshot SVD here uses the raw (unweighted) SVD,
exactly as for the state POD basis in `pod.py` -- the same cell-area-constant argument in that
module's docstring applies verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import SpectralGrid
from ..operators import dealias_mask
from ..rhs import vorticity_rhs
from .galerkin import spectral_partials, velocity_of


def build_nonlinear_snapshots(omega_snapshots: np.ndarray, grid: SpectralGrid,
                               dealias: bool = True) -> np.ndarray:
    """N(omega_j) = -(u.grad(omega))_j (physical space, flattened) for each snapshot -- the
    nonlinear-term analogue of the state snapshot matrix."""
    mask = dealias_mask(grid) if dealias else None
    cols = []
    for omega in omega_snapshots:
        omega_hat = grid.fft2(omega)
        n_hat = vorticity_rhs(omega_hat, grid, mask)
        cols.append(grid.ifft2_real(n_hat).reshape(-1))
    return np.array(cols).T  # (N_dof, M)


def compute_deim_basis(nonlinear_snapshots: np.ndarray, m: int) -> np.ndarray:
    """Leading m left singular vectors of the (raw) nonlinear snapshot matrix."""
    U, _, _ = np.linalg.svd(nonlinear_snapshots, full_matrices=False)
    return U[:, :m]


def select_deim_points(Psi: np.ndarray) -> np.ndarray:
    """Greedy DEIM interpolation-point selection (Chaturantabut & Sorensen 2010, Algorithm 1)."""
    m = Psi.shape[1]
    points = [int(np.argmax(np.abs(Psi[:, 0])))]
    for l in range(1, m):
        P = np.array(points)
        A = Psi[P, :l]
        b = Psi[P, l]
        c = np.linalg.solve(A, b)
        residual = Psi[:, l] - Psi[:, :l] @ c
        points.append(int(np.argmax(np.abs(residual))))
    return np.array(points, dtype=int)


@dataclass
class DEIMOperators:
    D: np.ndarray  # (r, m): Phi.T @ Psi @ inv(Psi[points, :])
    points: np.ndarray  # (m,) flat grid indices
    mode_u_pts: np.ndarray  # (r, m)
    mode_v_pts: np.ndarray
    mode_fx_pts: np.ndarray
    mode_fy_pts: np.ndarray


def build_deim_operators(Phi: np.ndarray, Psi: np.ndarray, points: np.ndarray,
                          grid: SpectralGrid) -> DEIMOperators:
    r = Phi.shape[1]
    ny, nx = grid.Ny, grid.Nx
    c = grid.dx * grid.dy  # cell area: Phi's columns are L2- (not Euclidean-) orthonormal (pod.py),
                            # so projecting through them needs the same c factor as pod.project/the
                            # Q-tensor construction in galerkin.py -- omitting it here was a real bug,
                            # caught because it produced an almost-exactly-constant (~1/c) scale error.
    Psi_P = Psi[points, :]
    U = Psi @ np.linalg.inv(Psi_P)  # (N_dof, m)
    D = c * (Phi.T @ U)  # (r, m)

    m = len(points)
    mode_u_pts = np.empty((r, m))
    mode_v_pts = np.empty((r, m))
    mode_fx_pts = np.empty((r, m))
    mode_fy_pts = np.empty((r, m))
    for i in range(r):
        phi_i = Phi[:, i].reshape(ny, nx)
        u_i, v_i = velocity_of(phi_i, grid)
        fx_i, fy_i = spectral_partials(phi_i, grid)
        mode_u_pts[i] = u_i.reshape(-1)[points]
        mode_v_pts[i] = v_i.reshape(-1)[points]
        mode_fx_pts[i] = fx_i.reshape(-1)[points]
        mode_fy_pts[i] = fy_i.reshape(-1)[points]

    return DEIMOperators(D=D, points=points, mode_u_pts=mode_u_pts, mode_v_pts=mode_v_pts,
                          mode_fx_pts=mode_fx_pts, mode_fy_pts=mode_fy_pts)


def deim_nonlinear_rhs(a: np.ndarray, deim_ops: DEIMOperators) -> np.ndarray:
    """DEIM-approximated projected nonlinear term D @ N(omega_r(a))|_points, O(r*m), no grid-sized
    operation. Sign convention matches galerkin.reduced_rhs (N = -advection)."""
    u_pts = deim_ops.mode_u_pts.T @ a
    v_pts = deim_ops.mode_v_pts.T @ a
    fx_pts = deim_ops.mode_fx_pts.T @ a
    fy_pts = deim_ops.mode_fy_pts.T @ a
    adv_pts = u_pts * fx_pts + v_pts * fy_pts
    return -(deim_ops.D @ adv_pts)


def deim_approximation_error(Psi: np.ndarray, points: np.ndarray,
                              nonlinear_snapshots: np.ndarray) -> np.ndarray:
    """Relative L2 (raw Euclidean -- the ratio is weight-invariant, see pod.py docstring)
    reconstruction error of the DEIM approximation applied to each nonlinear snapshot, for
    independent verification (section 12) before DEIM is trusted inside the ROM."""
    Psi_P = Psi[points, :]
    U = Psi @ np.linalg.inv(Psi_P)
    errs = []
    for j in range(nonlinear_snapshots.shape[1]):
        f = nonlinear_snapshots[:, j]
        f_approx = U @ f[points]
        den = np.linalg.norm(f)
        errs.append(np.linalg.norm(f - f_approx) / den if den > 0 else 0.0)
    return np.array(errs)
