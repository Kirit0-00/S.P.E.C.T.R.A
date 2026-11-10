"""
Spatial Domain Gray-Level Transformations.

Theoretical Foundation (Gonzalez & Woods, Chapter 3):
- Grayscale Luminance Conversion: I(x, y) = 0.299 R + 0.587 G + 0.114 B
- Logarithmic Transformation: s = c * ln(1 + r)
- Power-Law (Gamma) Transformation: s = c * r^gamma
"""

from typing import Union
import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to single-channel 8-bit grayscale luminance.

    Args:
        img: Input image of shape (H, W, 3) or (H, W), uint8 or float.

    Returns:
        Grayscale image of shape (H, W), dtype uint8 in [0, 255].
    """
    if img.ndim == 2:
        if img.dtype != np.uint8:
            return normalize_to_uint8(img)
        return img.copy()

    if img.shape[2] == 4:
        # RGBA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Standard ITU-R BT.601 conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def to_float64(img: np.ndarray) -> np.ndarray:
    """
    Normalize image values to float64 in range [0.0, 1.0].

    Args:
        img: Image of shape (H, W) or (H, W, C), uint8 or numeric.

    Returns:
        Array of shape (H, W) or (H, W, C), dtype float64 in [0.0, 1.0].
    """
    arr = img.astype(np.float64)
    max_val = np.max(arr)
    if max_val > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Min-max linear scaling of any 2D/3D array into uint8 range [0, 255].

    Args:
        img: Input array of shape (H, W) or (H, W, C).

    Returns:
        Scaled array of shape (H, W) or (H, W, C), dtype uint8.
    """
    min_v = np.min(img)
    max_v = np.max(img)
    if np.isclose(max_v, min_v):
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - min_v) / (max_v - min_v) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Apply logarithmic transformation: s = c * ln(1 + r).
    Compresses dynamic range to visualize low-intensity details.

    Args:
        img: Input array (H, W), float or uint8.
        c: Scaling constant.

    Returns:
        Log-transformed array (H, W), dtype float64.
    """
    r = to_float64(img)
    s = c * np.log1p(r)
    return s


def gamma_transform(img: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    Apply power-law (gamma) transformation: s = c * r^gamma.
    Gamma < 1 expands dark tones; Gamma > 1 expands bright tones.

    Args:
        img: Input array (H, W), float or uint8.
        gamma: Exponent parameter.
        c: Scaling constant.

    Returns:
        Gamma-transformed array (H, W), dtype float64 in [0.0, 1.0].
    """
    r = to_float64(img)
    s = c * np.power(r, gamma)
    return np.clip(s, 0.0, 1.0)
