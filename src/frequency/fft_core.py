"""
2D Fast Fourier Transform (FFT) Core Operations.

Theoretical Foundation (Gonzalez & Woods, Chapter 4):
- 2D Discrete Fourier Transform:
  F(u, v) = sum_{x=0}^{M-1} sum_{y=0}^{N-1} f(x, y) * exp(-j * 2*pi * (ux/M + vy/N))
- 2D Inverse Discrete Fourier Transform:
  f(x, y) = 1/(MN) * sum_{u=0}^{M-1} sum_{v=0}^{N-1} F(u, v) * exp(j * 2*pi * (ux/M + vy/N))
- Centering / Spectrum Shift:
  f(x, y) * (-1)^(x + y) <===> F(u - M/2, v - N/2)
- Logarithmic Magnitude Spectrum:
  S(u, v) = c * ln(1 + |F(u, v)|)
"""

import numpy as np
from scipy import fft as sp_fft
from src.spatial.transforms import to_grayscale, normalize_to_uint8


def compute_2d_fft(img: np.ndarray, pad_optimal: bool = True) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Compute centered 2D Fast Fourier Transform of a grayscale image.

    Args:
        img: Input array (H, W), uint8 or float.
        pad_optimal: If True, zero-pads dimensions to composite numbers for fast FFT.

    Returns:
        Tuple of (F_shifted complex128 array, original_shape (H, W)).
    """
    if img.ndim == 2:
        gray = img.astype(np.float64)
    else:
        gray = to_grayscale(img).astype(np.float64)
    orig_shape = gray.shape

    if pad_optimal:
        opt_h = sp_fft.next_fast_len(orig_shape[0])
        opt_w = sp_fft.next_fast_len(orig_shape[1])
        padded = np.zeros((opt_h, opt_w), dtype=np.float64)
        padded[: orig_shape[0], : orig_shape[1]] = gray
    else:
        padded = gray

    # 2D FFT and center zero-frequency component
    f_transform = np.fft.fft2(padded)
    f_shifted = np.fft.fftshift(f_transform)
    return f_shifted, orig_shape


def compute_2d_ifft(f_shifted: np.ndarray, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """
    Compute Inverse 2D FFT from centered complex frequency domain representation.

    Args:
        f_shifted: Centered complex spectrum array.
        target_shape: (H, W) to crop if padding was applied.

    Returns:
        Spatial domain real array, float64.
    """
    # De-shift and inverse FFT
    f_ishifted = np.fft.ifftshift(f_shifted)
    reconstructed = np.fft.ifft2(f_ishifted)
    real_part = np.real(reconstructed)

    if target_shape is not None:
        real_part = real_part[: target_shape[0], : target_shape[1]]

    return real_part


def get_magnitude_spectrum(f_shifted: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Compute the magnitude spectrum |F(u, v)|.

    Args:
        f_shifted: Centered complex frequency representation.
        log_scale: If True, applies s = ln(1 + |F(u, v)|) for visualization.

    Returns:
        Magnitude array (H, W), float64 normalized to [0.0, 1.0].
    """
    magnitude = np.abs(f_shifted)
    if log_scale:
        magnitude = np.log1p(magnitude)
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude /= max_val
    return magnitude


def get_phase_spectrum(f_shifted: np.ndarray) -> np.ndarray:
    """
    Compute phase angle spectrum: phi(u, v) = arctan2(Im(F), Re(F)).

    Returns:
        Phase spectrum in radians [-pi, pi], float64.
    """
    return np.angle(f_shifted)
