"""Manufactured-solution test of the nonlinear advection term.

Taylor-Green (test_taylor_green.py) has an identically zero nonlinear term
and so cannot detect a bug in the advection term. This test uses a
two-Fourier-mode manufactured solution (src/manufactured.py) with a
genuinely nonzero Jacobian, forced so the manufactured field is an exact
solution, and checks both pointwise accuracy and the expected 4th-order
temporal convergence rate of the RK4 stages that step the nonlinear term.
"""
import numpy as np

from src import SimParams, SpectralGrid, run, ManufacturedSolution


def _run_mms(N: int, dt: float, t_end: float, ms: ManufacturedSolution) -> float:
    grid = SpectralGrid(N, N)
    omega0 = ms.omega_exact(grid.X, grid.Y, 0.0)
    p = SimParams(Nx=N, Ny=N, nu=ms.nu, dt=dt, t_end=t_end, save_every=10**9, dealias=True)

    def forcing(t):
        return grid.fft2(ms.forcing(grid.X, grid.Y, t))

    res = run(p, omega0, forcing=forcing)
    exact_final = ms.omega_exact(grid.X, grid.Y, res.t[-1])
    return float(np.linalg.norm(res.omega[-1] - exact_final) / np.linalg.norm(exact_final))


def test_mms_matches_manufactured_solution():
    ms = ManufacturedSolution()
    err = _run_mms(N=64, dt=2e-3, t_end=0.1, ms=ms)
    assert err < 1e-8


def test_mms_temporal_convergence_is_fourth_order():
    ms = ManufacturedSolution()
    N = 64
    t_end = 0.2
    dts = [0.02, 0.01, 0.005]
    errs = [_run_mms(N=N, dt=dt, t_end=t_end, ms=ms) for dt in dts]

    orders = [
        np.log(errs[i - 1] / errs[i]) / np.log(dts[i - 1] / dts[i]) for i in range(1, len(dts))
    ]
    for order in orders:
        assert 3.7 < order < 4.3, f"observed order {order} not consistent with RK4"


def test_mms_resolved_case_reaches_machine_precision():
    """Once N fully resolves both manufactured wavenumbers (2/3-rule cutoff
    exceeds max wavenumber), spatial error should vanish to roundoff,
    leaving only the (separately tested) temporal error."""
    ms = ManufacturedSolution(k1x=3, k1y=4, k2x=6, k2y=2)
    err = _run_mms(N=32, dt=1e-4, t_end=0.01, ms=ms)
    assert err < 1e-9
