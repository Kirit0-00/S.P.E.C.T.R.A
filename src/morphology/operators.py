"""
Morphological Operators & Structuring Elements.

Theoretical Foundation (Gonzalez & Woods, Chapter 9.1 - 9.3):
- Structuring Elements (SE): Matrix defining neighborhood shape (Rectangle, Ellipse, Cross).
- Erosion: (A (-) B) = {z | (B)_z subseteq A} (Shrinks regions, eliminates noise spikes)
- Dilation: (A (+) B) = {z | (B^hat)_z cap A != empty} (Expands boundaries, fills voids)
- Opening: A circ B = (A (-) B) (+) B (Removes foreground protrusions and isolated dots)
- Closing: A bullet B = (A (+) B) (-) B (Welds gaps, fuses narrow breaks)
- Morphological Gradient: G = (A (+) B) - (A (-) B) (Edge enhancement)
"""

import cv2
import numpy as np


def get_structuring_element(shape: str = "rect", ksize: int = 5) -> np.ndarray:
    """
    Generate standard 2D morphological structuring element.

    Args:
        shape: 'rect', 'ellipse', or 'cross'.
        ksize: Dimension of kernel (positive odd integer).

    Returns:
        Structuring element binary kernel (ksize, ksize), uint8.
    """
    if ksize % 2 == 0:
        ksize += 1

    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    cv2_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(cv2_shape, (ksize, ksize))


def apply_erosion(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Apply morphological erosion: (A (-) B)."""
    kernel = get_structuring_element(shape, ksize)
    return cv2.erode(img, kernel, iterations=iterations)


def apply_dilation(img: np.ndarray, ksize: int = 3, shape: str = "rect", iterations: int = 1) -> np.ndarray:
    """Apply morphological dilation: (A (+) B)."""
    kernel = get_structuring_element(shape, ksize)
    return cv2.dilate(img, kernel, iterations=iterations)


def apply_opening(img: np.ndarray, ksize: int = 3, shape: str = "ellipse") -> np.ndarray:
    """Apply morphological opening: A circ B = (A (-) B) (+) B."""
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def apply_closing(img: np.ndarray, ksize: int = 7, shape: str = "ellipse") -> np.ndarray:
    """Apply morphological closing: A bullet B = (A (+) B) (-) B."""
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def apply_morphological_gradient(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply morphological gradient: G = (A (+) B) - (A (-) B)."""
    kernel = get_structuring_element("rect", ksize)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
