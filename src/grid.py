"""Doubly-periodic Fourier grid and wavenumber bookkeeping.

All fields are stored as real arrays of shape (Ny, Nx) in physical space and
as complex arrays of shape (Ny, Nx) in Fourier space, using numpy.fft.fft2 /
ifft2 (full complex transform, not the real-input rfft2 variant). This keeps
the wavenumber bookkeeping simple and easy to verify against hand
derivations; see README "Limitations" for the performance cost of this
choice relative to rfft2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralGrid:
    """Doubly-periodic grid on [0, Lx) x [0, Ly) with Nx x Ny points."""

    Nx: int
    Ny: int
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

        x = np.arange(self.Nx) * self.dx
        y = np.arange(self.Ny) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")  # shape (Ny, Nx)

        kx = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="xy")  # shape (Ny, Nx)

        self.K2 = self.KX**2 + self.KY**2
        self.K2_inv = np.zeros_like(self.K2)
        nonzero = self.K2 > 0
        self.K2_inv[nonzero] = 1.0 / self.K2[nonzero]

        # Nyquist wavenumbers must be zeroed when differentiating a
        # real-valued field spectrally (their formal derivative is not
        # representable by a real field on an even grid); harmless if not
        # exactly at Nyquist but included for correctness at any Nx, Ny.
        self.nyquist_mask = np.ones_like(self.KX)
        if self.Nx % 2 == 0:
            self.nyquist_mask[:, self.Nx // 2] = 0.0
        if self.Ny % 2 == 0:
            self.nyquist_mask[self.Ny // 2, :] = 0.0

    def fft2(self, f: np.ndarray) -> np.ndarray:
        return np.fft.fft2(f)

    def ifft2_real(self, f_hat: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(f_hat))
