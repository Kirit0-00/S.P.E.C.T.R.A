"""
Homomorphic Filtering for Illumination Inconsistency Detection.

Theoretical Foundation (Gonzalez & Woods, Chapter 4.9.6):
- Image Model: f(x, y) = i(x, y) * r(x, y)
  - i(x, y): Illumination (spatial variation slow -> low frequencies)
  - r(x, y): Reflectance (object details and seams -> high frequencies)
- Decoupling Pipeline:
  1. z(x, y) = ln(f(x, y) + eps)
  2. Z(u, v) = FFT{z(x, y)}
  3. S(u, v) = H(u, v) * Z(u, v)
     where H(u, v) = (gamma_H - gamma_L) * [1 - exp(-c * D(u, v)^2 / D_0^2)] + gamma_L
  4. s(x, y) = IFFT{S(u, v)}
  5. g(x, y) = exp(Re{s(x, y)}) - eps
- Forensic Application:
  Reveals divergent lighting vectors and sharp shadow/reflectance seams in spliced composites.
"""

import numpy as np
from src.frequency.fft_core import compute_2d_fft, compute_2d_ifft
from src.frequency.frequency_filters import compute_distance_matrix
from src.spatial.transforms import to_grayscale, normalize_to_uint8


def create_homomorphic_filter(
    shape: tuple[int, int],
    d0: float = 30.0,
    gamma_low: float = 0.5,
    gamma_high: float = 2.0,
    c: float = 1.0,
) -> np.ndarray:
    """
    Construct high-emphasis homomorphic transfer function:
        H(u, v) = (gamma_H - gamma_L) * [1 - exp(-c * D(u, v)^2 / D_0^2)] + gamma_L
    """
    d = compute_distance_matrix(shape)
    h = (gamma_high - gamma_low) * (1.0 - np.exp(-c * (d**2) / (d0**2))) + gamma_low
    return h


def apply_homomorphic_filter(
    img: np.ndarray,
    d0: float = 30.0,
    gamma_low: float = 0.5,
    gamma_high: float = 2.0,
    c: float = 1.0,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform complete homomorphic filtering on an image.

    Returns:
        Tuple of (enhanced_image, illumination_residual_map) both float64 in [0.0, 1.0].
    """
    gray = to_grayscale(img).astype(np.float64) / 255.0
    orig_shape = gray.shape

    # 1. Logarithmic transformation to decouple multiplicative components
    z = np.log(gray + eps)

    # 2. 2D FFT into frequency domain
    z_fft, pad_shape = compute_2d_fft(z, pad_optimal=True)

    # 3. Create and apply homomorphic transfer function
    h_filter = create_homomorphic_filter(
        z_fft.shape,
        d0=d0,
        gamma_low=gamma_low,
        gamma_high=gamma_high,
        c=c,
    )
    s_fft = z_fft * h_filter

    # 4. Inverse 2D FFT
    s_spatial = compute_2d_ifft(s_fft, target_shape=orig_shape)

    # 5. Exponential inversion
    enhanced = np.exp(s_spatial) - eps
    enhanced = np.clip(enhanced, 0.0, 1.0)

    # Illumination anomaly residual: difference between input luminance and reflectance
    residual = np.abs(gray - enhanced)
    max_res = np.max(residual)
    if max_res > 0:
        residual /= max_res

    return enhanced, residual
