"""Proper Orthogonal Decomposition (POD) of a vorticity-snapshot ensemble, via the singular value
decomposition (SVD).

## Inner product

Diagnostics elsewhere in this repository (`src.diagnostics.kinetic_energy`, `.enstrophy`) already use
the discrete $L^2$ inner product appropriate for this uniform, doubly-periodic grid: for fields
$f, g$ sampled at the $N_x \\times N_y$ grid points,

    <f, g> = (dx * dy) * sum_{grid points} f * g,

the rectangle-rule quadrature approximation of $\\int_\\Omega f g\\,dA$ -- spectrally accurate here
because the grid is uniform and periodic and the fields are smooth. POD is defined with respect to
THIS inner product, not the raw (unweighted) Euclidean dot product of flattened snapshot vectors, so
that captured "energy" and reconstruction error below are stated in the same physical units as
`kinetic_energy`/`enstrophy`.

## POD as an optimization problem, and its relation to the SVD

Given snapshots $\\{\\omega_j\\}_{j=1}^M$ (mean-subtracted if a mean/base state is used -- this project
does NOT mean-center, see the README for why), POD seeks the rank-$r$ subspace $V_r$ minimizing the
total (inner-product-weighted) projection error,

    min_{dim(V_r)=r}  sum_j || omega_j - P_{V_r} omega_j ||_W^2,   ||f||_W^2 := <f, f>.

Because the grid is UNIFORM, the inner-product weight is a single scalar $c = dx\\,dy$ times the
identity ($W = c I$ on the flattened vector space), rather than a general diagonal/mass matrix. This
is the key simplification exploited here: writing the (mean-subtracted) flattened snapshot matrix as
$X \\in \\mathbb{R}^{N_{\\mathrm{dof}}\\times M}$ and its ORDINARY (unweighted) economy SVD as
$X = U\\Sigma V^T$ ($U$, $V$ Euclidean-orthonormal), the solution of the weighted problem above is

    Phi_i = U[:, i] / sqrt(c),    i = 1..r,

which is straightforward to verify: $U$'s columns already solve the *unweighted* Eckart-Young
problem, and rescaling every column of an orthonormal matrix by the SAME constant $1/\\sqrt{c}$ turns
Euclidean-orthonormality ($U^T U = I$) into $W$-orthonormality ($\\Phi^T W \\Phi = c\\,\\Phi^T\\Phi =
c\\,(U/\\sqrt c)^T(U/\\sqrt c) = U^T U = I$), and a uniform positive rescaling of an orthonormal basis
does not change which subspace it spans, hence does not change the minimizer of the projection-error
functional above -- only the normalization of the individual mode vectors. (On a NON-uniform grid or
with a genuinely non-constant weight, this shortcut would not apply, and the "method of snapshots" --
solving the $M\\times M$ eigenproblem for the WEIGHTED Gram matrix $X^T W X$ -- would be needed
instead; `numpy.linalg.svd`'s economy mode already exploits the $M \\ll N_{\\mathrm{dof}}$ structure
internally in the same way the method of snapshots does, so no separate implementation of that
algorithm is required here.)

A useful, easily-overlooked consequence: because $c$ is a single global constant, it cancels exactly
out of every RATIO reported in this project (relative projection error, cumulative energy fraction),
so those numbers are numerically identical whether computed with the raw Euclidean SVD quantities or
the $W$-weighted ones. It does NOT cancel out of absolute quantities (mode normalization, reconstructed
energy/enstrophy from POD coefficients), which is why the rescaling above is still implemented and
tested explicitly (`tests/test_rom_pod.py`) rather than skipped as "not mattering".

## Singular values and captured energy

$$\\sum_{j=1}^M \\|\\omega_j\\|_W^2 = c \\sum_i \\sigma_i^2$$

so the fraction of total snapshot "energy" captured by the leading $r$ modes is
$\\sum_{i\\le r}\\sigma_i^2 \\big/ \\sum_i \\sigma_i^2$ -- the ordinary SVD cumulative-energy-fraction
formula, unaffected by the constant-weight rescaling (see above).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import SpectralGrid


def cell_area(grid: SpectralGrid) -> float:
    return grid.dx * grid.dy


def l2_inner_product(f: np.ndarray, g: np.ndarray, grid: SpectralGrid) -> float:
    """Discrete L2 inner product on the uniform periodic grid (see module docstring)."""
    return float(cell_area(grid) * np.sum(f * g))


def l2_norm(f: np.ndarray, grid: SpectralGrid) -> float:
    return float(np.sqrt(l2_inner_product(f, f, grid)))


@dataclass
class PODBasis:
    Phi: np.ndarray  # (N_dof, r): L2-orthonormal POD modes, flattened
    singular_values: np.ndarray  # (r,): sigma_i from the raw (unweighted) SVD of the snapshot matrix
    coefficients: np.ndarray  # (r, M): a_i(t_j) = <omega_j, Phi_i>_W for every training snapshot
    all_singular_values: np.ndarray  # (k,): full spectrum (k = min(N_dof, M)), for decay plots
    grid: SpectralGrid
    ny: int
    nx: int

    def mode_field(self, i: int) -> np.ndarray:
        return self.Phi[:, i].reshape(self.ny, self.nx)


def compute_pod_basis(omega_snapshots: np.ndarray, grid: SpectralGrid, r: int) -> PODBasis:
    """omega_snapshots: (M, Ny, Nx) physical-space snapshots (NOT mean-subtracted; this project uses
    pure, non-mean-centered POD modes -- see module docstring / README). Returns the leading r POD
    modes, L2-orthonormal, ranked by captured variance."""
    ny, nx = omega_snapshots.shape[1:]
    X = omega_snapshots.reshape(omega_snapshots.shape[0], -1).T  # (N_dof, M)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(r, U.shape[1])
    c = cell_area(grid)

    Phi = U[:, :r] / np.sqrt(c)
    coefficients = np.sqrt(c) * (S[:r, None] * Vt[:r, :])  # (r, M)

    return PODBasis(Phi=Phi, singular_values=S[:r], coefficients=coefficients,
                     all_singular_values=S, grid=grid, ny=ny, nx=nx)


def cumulative_energy_fraction(singular_values: np.ndarray, r: int | None = None) -> np.ndarray:
    """Cumulative sum_{i<=r} sigma_i^2 / sum_all sigma_i^2, for r = 1..len(singular_values) (or up to
    the given r). Uses the FULL spectrum in the denominator when available (pass
    `basis.all_singular_values` for the true total, not just the retained-r subset)."""
    energy = singular_values ** 2
    cum = np.cumsum(energy) / np.sum(energy)
    return cum if r is None else cum[:r]


def project(field: np.ndarray, Phi: np.ndarray, grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray]:
    """Project a single physical-space field (Ny, Nx) onto the POD basis Phi (N_dof, r). Returns
    (coefficients (r,), reconstructed field (Ny, Nx))."""
    ny, nx = field.shape
    f_flat = field.reshape(-1)
    c = cell_area(grid)
    a = c * (Phi.T @ f_flat)  # a_i = <f, Phi_i>_W = c * f . Phi_i
    recon = (Phi @ a).reshape(ny, nx)
    return a, recon


def projection_error(field: np.ndarray, Phi: np.ndarray, grid: SpectralGrid) -> float:
    """Relative L2 (W-weighted) projection error ||f - P_r f|| / ||f||."""
    _, recon = project(field, Phi, grid)
    num = l2_norm(field - recon, grid)
    den = l2_norm(field, grid)
    return num / den if den > 0 else 0.0
