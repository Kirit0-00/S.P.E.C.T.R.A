"""
Spatial Domain Linear & Non-Linear Filtering.

Theoretical Foundation (Gonzalez & Woods, Chapter 3.4 - 3.6):
- Low-Pass Smoothing: 2D Gaussian Kernel G(x, y) = 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))
- Edge-Preserving Smoothing: Bilateral Filter combining spatial and radiometric distances
- High-Pass Edge Detection:
  - Sobel Gradients: G_x, G_y and magnitude M(x, y) = sqrt(G_x^2 + G_y^2)
  - 2D Laplacian operator: nabla^2 f = d^2f/dx^2 + d^2f/dy^2
  - Unsharp Masking / High-Boost: f_hb(x, y) = f(x, y) + k * (f(x, y) - f_lp(x, y))
- Local Statistical Variance: sigma^2(x, y) = E[I^2] - (E[I])^2
"""

import cv2
import numpy as np
from src.spatial.transforms import to_grayscale, normalize_to_uint8


def apply_gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply low-pass Gaussian smoothing filter.

    Args:
        img: Input array (H, W) or (H, W, C), uint8.
        ksize: Aperture size (must be positive odd integer).
        sigma: Gaussian kernel standard deviation.

    Returns:
        Smoothed array of identical shape and dtype.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_bilateral_filter(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Apply edge-preserving bilateral smoothing.
    Eliminates micro-texture while preserving sharp tampering boundaries.
    """
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def compute_sobel_gradients(img: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1st order directional Sobel spatial derivatives.

    Returns:
        Tuple of (G_x, G_y, Magnitude) all float64 in [0.0, 1.0].
    """
    gray = to_grayscale(img).astype(np.float64) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx**2 + gy**2)
    max_m = np.max(magnitude)
    if max_m > 0:
        magnitude /= max_m
    return gx, gy, magnitude


def apply_laplacian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply 2nd order isotropic Laplacian derivative filter: nabla^2 f.
    Highlights sharp spatial discontinuities and spliced seams.

    Returns:
        Laplacian response, float64 normalized to [0.0, 1.0].
    """
    gray = to_grayscale(img).astype(np.float64) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    lap_abs = np.abs(lap)
    max_val = np.max(lap_abs)
    if max_val > 0:
        lap_abs /= max_val
    return lap_abs


def apply_unsharp_mask(img: np.ndarray, ksize: int = 5, sigma: float = 1.0, k: float = 1.5) -> np.ndarray:
    """
    Apply high-boost unsharp masking: f_hb = f + k * (f - f_smooth).

    Args:
        img: Input array (H, W) or (H, W, 3), uint8.
        ksize: Low-pass filter kernel size.
        sigma: Low-pass filter standard deviation.
        k: High-boost amplification factor (k=1 is standard unsharp mask).

    Returns:
        Sharpened array, uint8 in [0, 255].
    """
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    f = img.astype(np.float64)
    b = blurred.astype(np.float64)
    mask = f - b
    sharpened = f + k * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def compute_local_variance(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Compute local statistical variance: sigma^2(x, y) = E[I^2] - (E[I])^2.
    Used for flat-region gating to eliminate false positives in feature matching.

    Returns:
        Local variance map (H, W), float64.
    """
    gray = to_grayscale(img).astype(np.float64)
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    mean = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REFLECT)
    sqr_mean = cv2.filter2D(gray**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(sqr_mean - mean**2, 0.0)
    return var
