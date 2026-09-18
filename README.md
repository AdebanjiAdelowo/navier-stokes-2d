# 2D Incompressible Navier-Stokes: Vorticity-Streamfunction Pseudo-Spectral Solver

A verified pseudo-spectral solver for the 2D incompressible Navier-Stokes equations on a doubly
periodic domain, validated against the Taylor-Green vortex exact solution and a symbolically
derived manufactured solution, with a second benchmark (freely decaying 2D turbulence) assessed
through physical conservation diagnostics and spatial self-convergence.

## Overview

This project implements and validates a classical numerical method for 2D incompressible flow
rather than proposing a new one. The goal is to demonstrate the complete workflow expected of a
credible numerical-PDE implementation: formulate the equations, choose and justify a discretisation,
implement it as reusable code, verify it against problems with known answers (an exact solution and
a manufactured solution), quantify convergence rates, check physical conservation laws, and only
then use the solver for a genuine flow-physics experiment.

## Mathematical problem

The 2D incompressible Navier-Stokes equations, in velocity-pressure form, are

$$\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \nu\nabla^2 \mathbf{u}, \qquad \nabla\cdot\mathbf{u} = 0,$$

on the doubly periodic domain $[0, L_x)\times[0, L_y)$. In two dimensions it is standard, and
mathematically convenient, to work instead with the scalar vorticity
$\omega = \partial_x v - \partial_y u$ and a streamfunction $\psi$ satisfying $\nabla\cdot\mathbf{u}=0$
automatically. Taking the curl of the momentum equation eliminates the pressure entirely, giving a
single scalar transport equation:

$$\partial_t \omega + \mathbf{u}\cdot\nabla\omega = \nu\nabla^2\omega, \qquad \mathbf{u} = (\partial_y\psi,\, -\partial_x\psi), \qquad -\nabla^2\psi = \omega.$$

## Governing equations and numerical method

**Why vorticity-streamfunction, and why pseudo-spectral.** On a periodic domain with no solid
boundaries, the vorticity-streamfunction formulation removes both the pressure (a Poisson equation
that would otherwise have to be solved for $p$) and the incompressibility constraint (satisfied
identically by construction, not enforced iteratively as in a velocity-pressure projection method).
What remains is a single scalar advection-diffusion-type equation plus one Poisson solve per time
step. On a periodic domain, the Fourier basis diagonalises every differential operator that appears
(gradient, Laplacian, and hence the Poisson solve), so a Fourier pseudo-spectral discretisation turns
every linear operation into an elementwise multiplication and gives spectral (faster than any fixed
polynomial order) accuracy in space for smooth solutions, at the cost of $O(N^2\log N)$ FFTs per
evaluation of the right-hand side. This is the standard method of choice for periodic 2D
incompressible flow and turbulence studies (Orszag, 1971; Canuto et al., 2006).

**Discretisation, term by term** (implementation in `src/`):

- *Fourier differentiation and the streamfunction Poisson equation* (`src/grid.py`,
  `src/operators.py`): on an $N_x\times N_y$ grid, wavenumbers $(k_x, k_y)$ are obtained from
  `numpy.fft.fftfreq`; any derivative $\partial_x$ becomes multiplication by $ik_x$ in Fourier space.
  The Poisson equation $-\nabla^2\psi=\omega$ becomes $\hat\psi = \hat\omega / (k_x^2+k_y^2)$,
  elementwise, in $O(N^2\log N)$ via FFT.
- *The zero Fourier mode*: $k=0$ makes the Poisson divisor singular. This is the usual
  compatibility condition for a periodic Poisson equation: $\psi$ is only defined up to an additive
  constant, which does not affect $\mathbf{u}=\nabla^\perp\psi$, so $\hat\psi(0,0)$ is set to zero by
  convention (`operators.poisson_solve`).
- *Velocity recovery*: $\hat u = ik_y\hat\psi$, $\hat v = -ik_x\hat\psi$. Because both components come
  from derivatives of the same scalar $\psi$, $\nabla\cdot\mathbf{u} = \partial_x u + \partial_y v =
  -k_xk_y\hat\psi + k_xk_y\hat\psi \equiv 0$ **identically**, to machine precision, by construction
  (verified in `tests/test_operators.py::test_velocity_from_psi_is_divergence_free` and tracked
  through full runs by the `max_divergence` diagnostic).
- *Nonlinear advection* (`src/rhs.py`): $\mathbf{u}\cdot\nabla\omega$ is evaluated pseudo-spectrally
  — $u, v, \omega_x, \omega_y$ are computed in Fourier space and transformed to physical space by
  inverse FFT, multiplied pointwise, and transformed back. This quadratic product aliases energy from
  wavenumber pairs whose sum exceeds the resolvable range back into the resolved range; the standard
  2/3-rule filter (Orszag, 1971), applied in `operators.dealias_mask` and `rhs.vorticity_rhs`, removes
  this exactly for a single quadratic nonlinearity by truncating each direction to its lowest 2/3 of
  wavenumbers before the product is formed.
- *Time integration* (`src/solver.py`): the viscous term $\nu\nabla^2\omega$ is diagonal in Fourier
  space and stiff at high wavenumber, so it is treated **exactly** via an integrating factor rather
  than explicitly; the nonlinear term is stepped with classical explicit RK4 applied to the
  integrating-factor-transformed variable (the "integrating factor RK4" scheme, e.g. Trefethen, 2000;
  Canuto et al., 2006). The exact stage formulas, derived from first principles, are documented in the
  `solver.py` module docstring. Because diffusion is integrated exactly, the time-step restriction
  comes only from the explicit advection term, not from viscosity, however fine the grid.
- *Stability*: an empirical check (inviscid, $N=128$, broadband initial condition; see "Verification
  and validation") found the solver stable up to an advective CFL number $U_{\max}\Delta t/\Delta x
  \approx 1.5$ and unstable by $\approx 2.3$, consistent with the finite stability region of explicit
  RK4 applied to the (purely imaginary, advection-dominated) eigenvalues of the nonlinear term. All
  benchmark runs in this repository use CFL numbers well under 0.1.

## Implementation

Reusable package under `src/`, not a notebook:

```
src/
├── grid.py            SpectralGrid: wavenumbers, FFT/IFFT wrappers, Nyquist handling
├── params.py           SimParams: all physical/numerical parameters in one dataclass
├── operators.py         poisson_solve, velocity_from_psi, dealias_mask
├── rhs.py               vorticity_rhs: pseudo-spectral nonlinear advection (+ optional forcing)
├── solver.py             run(): integrating-factor RK4 time integration
├── diagnostics.py        kinetic_energy, enstrophy, divergence_l2, max_divergence
├── exact.py               Taylor-Green vortex exact solution
├── manufactured.py         sympy-derived two-mode manufactured solution + forcing
└── initial_conditions.py    reproducible random vorticity field for the turbulence benchmark
```

## Verification and validation

Every claim below is backed by a script or test in this repository; none is asserted without a
corresponding run. 21 automated tests (`tests/`) check correctness before any of the results below
are trusted (`pytest tests/ -v`, all passing at time of writing).

**Operator-level checks** (`tests/test_operators.py`): the Poisson solve and velocity recovery are
checked against a hand-derived single-Fourier-mode manufactured solution (exact to $10^{-10}$); the
divergence-free identity is checked on a random field; the dealiasing mask and Nyquist handling are
checked structurally.

**Taylor-Green vortex** (`src/exact.py`, `tests/test_taylor_green.py`,
`examples/taylor_green.py`): on $[0,2\pi)^2$, $u=\cos(kx)\sin(ky)e^{-2\nu k^2t}$,
$v=-\sin(kx)\cos(ky)e^{-2\nu k^2 t}$ is an **exact** solution of 2D Navier-Stokes, with vorticity
$\omega=-2k\cos(kx)\cos(ky)e^{-2\nu k^2t}$. A property of this specific flow, verified directly in
`test_taylor_green_nonlinear_term_is_zero`, is that $\mathbf{u}\cdot\nabla\omega\equiv 0$: because
$\omega$ is proportional to a single Laplacian eigenfunction of $\psi$, the Jacobian $J(\psi,\omega)
= cJ(\psi,\psi) = 0$ for any constant $c$. Vorticity therefore obeys pure linear diffusion, which the
integrating factor solves exactly. Consequently, at $N=64$, $\nu=0.05$, $k=2$, $t_{\mathrm{end}}=0.3$:
**relative $L^2$ error $= 7.03\times10^{-15}$**, i.e. floating-point roundoff, not a decaying
discretisation error. This is an excellent check of the linear machinery (spectral derivatives,
Poisson solve, diffusion/time-integration), but by itself it **cannot** detect a bug in the nonlinear
advection term, since that term evaluates to zero either way.

**Manufactured solution for the nonlinear term** (`src/manufactured.py`,
`scripts/derive_mms_forcing.py`, `tests/test_mms.py`): to close that gap, a streamfunction built from
the **sum of two** non-parallel Fourier modes is used, so $\omega$ is not proportional to $\psi$ and
the Jacobian is genuinely nonzero. The forcing term required to make this an exact solution of the
forced vorticity equation is derived symbolically with `sympy` (method of manufactured solutions;
Roache, 2002) rather than by hand, to eliminate transcription error, and evaluated via
`sympy.lambdify`. At $N=64$, $\Delta t=2\times10^{-3}$, $t_{\mathrm{end}}=0.1$: **relative $L^2$
error $=3.3\times10^{-15}$**. A temporal convergence sweep at fixed, fully resolved $N=64$
($\Delta t \in \{0.02, 0.01, 0.005, 0.0025, 0.00125\}$) gives:

| $\Delta t$ | relative $L^2$ error | observed order |
|---:|---:|---:|
| 0.02000 | $5.16\times10^{-11}$ | — |
| 0.01000 | $3.23\times10^{-12}$ | 4.000 |
| 0.00500 | $2.02\times10^{-13}$ | 3.997 |
| 0.00250 | $1.36\times10^{-14}$ | 3.890 |
| 0.00125 | $4.48\times10^{-15}$ | 1.606 |

confirming clean **4th-order** convergence of the RK4 nonlinear-term integration down to the point
where the error reaches the double-precision floor ($\sim10^{-15}$), after which the observed order
degrades because roundoff, not truncation error, dominates — expected behaviour, reported honestly
rather than cut off at a flattering point. Reproduce with `python scripts/convergence_temporal.py`
(figure: `figures/convergence_temporal.png`).

**Spatial resolution** (`scripts/convergence_spatial.py`): a pseudo-spectral method has no fixed
polynomial spatial order for smooth periodic data. For a **band-limited** manufactured solution
(wavenumbers up to $k=6$), once $N$ is large enough that the 2/3-dealiasing cutoff exceeds every
wavenumber present, the field is represented exactly and the error collapses to machine precision;
below that threshold the field is under-resolved and aliased, which does not produce a smooth
power-law curve (aliasing folds energy unpredictably depending on $N$):

| $N$ | 2/3-rule cutoff | relative $L^2$ error |
|---:|---:|---:|
| 8 | 2.67 | $2.79\times10^{-2}$ |
| 12 | 4.00 | $2.29\times10^{-2}$ |
| 16 | 5.33 | $6.28\times10^{-2}$ |
| 20 | 6.67 | $4.70\times10^{-2}$ |
| 24 | 8.00 | $4.70\times10^{-2}$ |
| 32 | 10.67 | $7.02\times10^{-15}$ |
| 48 | 16.00 | $6.38\times10^{-15}$ |

This is expected, correct spectral-method behaviour (figure: `figures/convergence_spatial_mms.png`),
not a bug, and it is reported as such rather than dressed up as an algebraic convergence rate that
would not be meaningful here. For a field that is **not** exactly band-limited, see the self-
convergence study below.

**Physical diagnostics** (`src/diagnostics.py`, `tests/test_diagnostics.py`): for unforced periodic
2D Navier-Stokes, advection conserves kinetic energy exactly and only viscosity dissipates it, giving
the exact identity $dE/dt = -2\nu\Omega(t)$ (energy budget). On a broadband random vorticity field
($N=48$, $\nu=0.02$), a central-difference estimate of $dE/dt$ from saved snapshots matches
$-2\nu\Omega(t)$ to a maximum relative mismatch of $1.3\times10^{-5}$ (interior points); with $\nu=0$
(inviscid), energy drifts by at most $10^{-3}$ relative over the tested short integration. Divergence
of the recovered velocity field is checked through every saved snapshot of every test run and stays
below $10^{-9}$ (typically $10^{-17}$, i.e. exactly zero up to roundoff), consistent with the
identically-zero divergence derived above.

## Experiments

### 1. Taylor-Green vortex (exact-solution validation)

`python examples/taylor_green.py` — $N=64$, $\nu=0.05$, $k=2$, $t_{\mathrm{end}}=0.3$. Produces
`figures/taylor_green_validation.png` (initial field, final numerical field, pointwise error).

### 2. Decaying 2D turbulence (second flow benchmark)

There is no closed-form solution for freely decaying 2D turbulence, so this benchmark is validated
through physical diagnostics rather than a pointwise comparison. The initial vorticity is an
isotropic random field with Fourier amplitude shaped by a bump function peaked at wavenumber $k_0$
(`src/initial_conditions.py`; energy concentrated at intermediate scales, the standard qualitative
setup for this class of problem, e.g. McWilliams, 1984 — the specific envelope used is our own
simple choice, not a reproduction of any particular paper's exact spectrum).

`python examples/decaying_turbulence.py --config local` — $N=128$, $\nu=0.01$, $k_0=8$,
$t_{\mathrm{end}}=4$ (runtime: 6.0 s on the benchmark hardware below):

- Kinetic energy decays **monotonically**, $E: 0.01549\to0.00123$ (92% loss).
- Enstrophy decays from $1.000\to0.0219$.
- Energy budget $dE/dt$ vs. $-2\nu\Omega(t)$: maximum relative mismatch $3.08\times10^{-4}$ over the
  full run (figure: `figures/decaying_turbulence_diagnostics_local.png`).
- $\max|\nabla\cdot\mathbf{u}|$ over the run: $1.34\times10^{-17}$.
- Vorticity snapshots (figure: `figures/decaying_turbulence_snapshots_local.png`) show the
  qualitatively expected coarsening: small-scale vortices merge into fewer, larger, longer-lived
  structures as the flow decays — the well-known inverse-cascade-like phenomenology of 2D turbulence,
  shown here as a qualitative observation, not a quantitatively fitted scaling law.

Three resolution presets are provided (`configs/{smoke,local,full}.yaml`): `smoke` ($N=48$,
$t_{\mathrm{end}}=0.2$, <1 s) for fast sanity checks, `local` (above) as the default, and `full`
($N=256$, $t_{\mathrm{end}}=6$) for a longer, higher-resolution run. None of these is silently
substituted for another; the config used is always named in the output filenames and printed results.

### 3. Spatial self-convergence of the turbulence benchmark

Because the turbulence initial condition is broadband, not band-limited, `scripts/convergence_spatial.py`
(Part B) runs it at $N\in\{16,24,32,48,64,96\}$ against a $N=256$ reference (spectral restriction of
the same physical initial condition to each grid; the restriction operator is unit-tested against a
resolved analytical field to $2\times10^{-15}$), all at $\nu=0.02$, $t_{\mathrm{end}}=0.1$:

| $N$ | relative $L^2$ error vs. $N=256$ reference |
|---:|---:|
| 16 | $6.47\times10^{-3}$ |
| 24 | $6.04\times10^{-3}$ |
| 32 | $4.43\times10^{-3}$ |
| 48 | $5.85\times10^{-4}$ |
| 64 | $4.97\times10^{-5}$ |
| 96 | $3.19\times10^{-8}$ |

a smooth, rapidly decreasing (spectral) convergence curve, as expected once the grid starts to
resolve the energy-containing scales near $k_0=6$ (figure: `figures/convergence_spatial_turbulence.png`).

## Error / convergence analysis

Summarised from "Verification and validation" above: **4th-order** temporal convergence confirmed
cleanly across three successive step-size halvings before hitting the double-precision floor;
**spectral (exponential)** spatial convergence confirmed for both a band-limited manufactured
solution (sharp resolution threshold) and a broadband turbulence field (smooth decay against a
high-resolution reference); exact ($<10^{-9}$, typically $10^{-17}$) incompressibility maintained
throughout; exact energy-budget identity reproduced to $10^{-4}$–$10^{-5}$ relative accuracy.

## Computational performance

Measured on: Apple M3 Pro, macOS-26.6.2-arm64, Python 3.12.2, NumPy 2.0.0, float64 / complex128,
single-threaded `numpy.fft` (CPU only, no GPU). 5 warm-up steps discarded; each entry is the median
of 3 repeats of 30 timed steps (`python scripts/benchmark_runtime.py`, full output in
`results/runtime_benchmark.txt`):

| $N$ | median ms/step | min | max |
|---:|---:|---:|---:|
| 32 | 0.420 | 0.419 | 0.424 |
| 64 | 0.817 | 0.801 | 0.819 |
| 128 | 2.626 | 2.623 | 2.633 |
| 256 | 12.603 | 11.575 | 12.735 |

Consistent with the expected $O(N^2\log N)$ pseudo-spectral cost. No claim of GPU acceleration,
multi-threading, or "real-time" performance is made; none was tested.

## Limitations

- **Full complex `fft2`/`ifft2`, not `rfft2`/`irfft2`.** All fields are real-valued, so a real-input
  FFT would roughly halve memory and FFT cost. `fft2` was chosen for simplicity of wavenumber
  bookkeeping and ease of verification; switching to `rfft2` is a natural, low-risk performance
  follow-up, not yet done.
- **Explicit treatment of advection limits the time step** (empirically, CFL $\lesssim 1.5$–2.3 for
  this RK4 scheme; see "Governing equations"), even though diffusion itself is unconditionally
  stable. This was characterised on one resolution/initial-condition combination, not swept
  systematically across $\nu$, $N$, and flow type.
- **Periodic boundary conditions only.** Wall-bounded flows (e.g. the classical cylinder-wake
  benchmark) require a different solver; see the companion FEM project planned for that case.
- **Decaying-turbulence initial spectrum is not matched to a specific literature reference**, so the
  turbulence benchmark's snapshots and decay curves are validated as *physically consistent*
  (conservation laws, self-convergence), not as a quantitative reproduction of a published turbulence
  statistic (e.g. a specific inverse-cascade slope).
- **Single random seed** for the turbulence benchmark; no ensemble averaging over initial conditions.
- CPU-only; no distributed- or GPU-computing was implemented or benchmarked.

## Reproducibility

All runs use explicit, fixed random seeds (`numpy.random.default_rng(seed)`) where randomness is
involved (`src/initial_conditions.py`), and `tests/test_solver.py::test_run_is_deterministic` checks
bit-for-bit reproducibility of the time stepper. No GPU-nondeterminism is possible since everything
runs on CPU in double precision.

## Repository structure

```
navier-stokes-2d/
├── README.md
├── requirements.txt
├── src/                  solver package (see "Implementation")
├── tests/                21 pytest tests: operators, exact solution, MMS, diagnostics, solver
├── examples/              taylor_green.py, decaying_turbulence.py
├── scripts/                derive_mms_forcing.py, convergence_temporal.py, convergence_spatial.py,
│                            benchmark_runtime.py
├── configs/                smoke.yaml, local.yaml, full.yaml (decaying-turbulence resolution presets)
├── figures/                 generated PNGs (validation, convergence, turbulence diagnostics)
└── results/                  generated text/NPZ outputs backing every number quoted above
```

## Installation

Requires Python $\geq 3.10$ (tested on 3.12). Dependencies: `numpy`, `scipy`, `matplotlib`, `pytest`,
`sympy` (manufactured-solution derivation only, not needed by the solver itself at run time beyond
the manufactured-solution module), `pyyaml` (config files).

```bash
cd navier-stokes-2d
pip install -r requirements.txt
```

## Running the experiments

```bash
# tests (run first; all downstream results assume these pass)
pytest tests/ -v

# Taylor-Green vortex validation
python examples/taylor_green.py

# decaying 2D turbulence (smoke / local / full)
python examples/decaying_turbulence.py --config smoke
python examples/decaying_turbulence.py --config local
python examples/decaying_turbulence.py --config full

# convergence studies
python scripts/convergence_temporal.py
python scripts/convergence_spatial.py

# runtime benchmark
python scripts/benchmark_runtime.py

# (optional) regenerate the manufactured-solution forcing derivation
python scripts/derive_mms_forcing.py
```

## References

- Orszag, S. A. (1971). "On the elimination of aliasing in finite-difference schemes by filtering
  high-wavenumber components." *J. Atmos. Sci.* 28(6), 1074.
- Canuto, C., Hussaini, M. Y., Quarteroni, A., Zang, T. A. (2006). *Spectral Methods: Fundamentals in
  Single Domains*. Springer.
- Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM.
- Taylor, G. I., Green, A. E. (1937). "Mechanism of the production of small eddies from large ones."
  *Proc. R. Soc. Lond. A* 158(895), 499-521.
- Roache, P. J. (2002). "Code Verification by the Method of Manufactured Solutions." *J. Fluids Eng.*
  124(1), 4-10.
- McWilliams, J. C. (1984). "The emergence of isolated coherent vortices in turbulent flow."
  *J. Fluid Mech.* 146, 21-43. (Cited for general context on decaying-2D-turbulence initial
  conditions; the specific spectral envelope used in this repository is not a reproduction of this
  paper's formula.)

**Prior related work.** The spectral/FFT/Leray-projection methodology used here builds on
methodological experience from an earlier project, `Master-Thesis` (Udine MSc thesis, Lin-Thiffeault-
Doering optimal-mixing solver, pseudo-spectral with Leray projection). That code was not reused here;
this solver was implemented independently, targeting the (different) Navier-Stokes vorticity-
transport problem and its own validation suite.
