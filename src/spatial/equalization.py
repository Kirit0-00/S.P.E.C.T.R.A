"""
Spatial Domain Histogram Equalization and Contrast Enhancement.

Theoretical Foundation (Gonzalez & Woods, Chapter 3.3):
- Global Histogram Equalization: Transformation T(r_k) = (L - 1) * sum_{j=0}^k p_r(r_j)
- CLAHE (Contrast-Limited Adaptive Histogram Equalization):
  Tiles image into contextual regions, clips local histogram at threshold beta,
  redistributes excess equally across bins, and uses bilinear interpolation
  to eliminate artificial boundary artifacts.
"""

import cv2
import numpy as np
from src.spatial.transforms import to_grayscale, normalize_to_uint8


def global_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Apply classical Global Histogram Equalization to spread intensity distributions.

    Args:
        img: Input image (H, W) or (H, W, 3), uint8.

    Returns:
        Equalized image (H, W) or (H, W, 3), dtype uint8.
    """
    if img.ndim == 2:
        gray = img if img.dtype == np.uint8 else normalize_to_uint8(img)
        return cv2.equalizeHist(gray)

    # For color images: convert to YCrCb and equalize only the Y (luminance) channel
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Exposes faint edge tampering and boundary artifacts without over-amplifying noise.

    Args:
        img: Input image (H, W) or (H, W, 3), uint8.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Number of contextual tiles (grid rows, cols).

    Returns:
        CLAHE-enhanced image, same shape and dtype uint8.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if img.ndim == 2:
        gray = img if img.dtype == np.uint8 else normalize_to_uint8(img)
        return clahe.apply(gray)

    # For 3-channel BGR: enhance luminance channel in YCrCb or LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compute_normalized_histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute normalized probability mass function (histogram) p_r(r_k).

    Args:
        gray: Grayscale image (H, W), uint8.
        bins: Number of intensity bins (default 256).

    Returns:
        1D array of shape (bins,), float64 summing to 1.0.
    """
    if gray.ndim != 2:
        gray = to_grayscale(gray)
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    total_pixels = gray.size
    return (hist / total_pixels).astype(np.float64)
