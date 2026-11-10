"""
Frequency Domain Filters (LPF and HPF).

Theoretical Foundation (Gonzalez & Woods, Chapter 4.8 - 4.9):
- Center-to-point Euclidean distance: D(u, v) = sqrt((u - M/2)^2 + (v - N/2)^2)
- Ideal Filters:
  - ILPF: H(u, v) = 1 if D(u, v) <= D_0 else 0
  - IHPF: 1 - ILPF
- Gaussian Filters:
  - GLPF: H(u, v) = exp(-D(u, v)^2 / (2 * D_0^2))
  - GHPF: 1 - GLPF
- Butterworth Filters (order n):
  - BLPF: H(u, v) = 1 / (1 + [D(u, v) / D_0]^(2n))
  - BHPF: H(u, v) = 1 / (1 + [D_0 / D(u, v)]^(2n))
"""

import numpy as np


def compute_distance_matrix(shape: tuple[int, int]) -> np.ndarray:
    """
    Compute Euclidean distance from each frequency point to the center (M/2, N/2).

    Args:
        shape: Dimensions (M, N).

    Returns:
        D(u, v) array of shape (M, N), float64.
    """
    m, n = shape
    u = np.arange(m, dtype=np.float64) - m / 2.0
    v = np.arange(n, dtype=np.float64) - n / 2.0
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(uu**2 + vv**2)


def create_gaussian_filter(shape: tuple[int, int], d0: float, high_pass: bool = False) -> np.ndarray:
    """
    Create a Gaussian Low-Pass or High-Pass filter transfer function.
    """
    d = compute_distance_matrix(shape)
    glpf = np.exp(-(d**2) / (2.0 * (d0**2)))
    return 1.0 - glpf if high_pass else glpf


def create_butterworth_filter(
    shape: tuple[int, int],
    d0: float,
    order: int = 2,
    high_pass: bool = False,
) -> np.ndarray:
    """
    Create a Butterworth Low-Pass or High-Pass filter transfer function.
    """
    d = compute_distance_matrix(shape)
    # Avoid zero-division at origin for high-pass
    eps = 1e-10
    if high_pass:
        h = 1.0 / (1.0 + np.power(d0 / (d + eps), 2 * order))
    else:
        h = 1.0 / (1.0 + np.power(d / d0, 2 * order))
    return h


def create_ideal_filter(shape: tuple[int, int], d0: float, high_pass: bool = False) -> np.ndarray:
    """
    Create an Ideal Low-Pass or High-Pass filter transfer function.
    """
    d = compute_distance_matrix(shape)
    ilpf = (d <= d0).astype(np.float64)
    return 1.0 - ilpf if high_pass else ilpf


def apply_frequency_filter(f_shifted: np.ndarray, h_filter: np.ndarray) -> np.ndarray:
    """
    Apply transfer function H(u, v) to centered Fourier representation:
        G(u, v) = H(u, v) * F(u, v)

    Returns:
        Filtered centered complex spectrum.
    """
    return f_shifted * h_filter
