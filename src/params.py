"""Simulation parameters for the 2D vorticity-streamfunction pseudo-spectral solver."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimParams:
    Nx: int = 64
    Ny: int = 64
    Lx: float = 6.283185307179586  # 2*pi
    Ly: float = 6.283185307179586
    nu: float = 1.0e-2
    dt: float = 1.0e-3
    t_end: float = 1.0
    dealias: bool = True
    save_every: int = 1  # store a snapshot every N steps (1 = every step)
