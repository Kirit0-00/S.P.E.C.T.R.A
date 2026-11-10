"""
Morphological Reconstruction & Boundary Refinement.

Theoretical Foundation (Gonzalez & Woods, Chapter 9.5):
- Geodesic Morphological Reconstruction: Repeated conditional dilation/erosion of marker inside mask.
- Morphological Hole Filling:
  X_k = (X_{k-1} (+) B) cap A^c, until X_k = X_{k-1}
  Seals internal voids within tampered clusters without expanding external perimeter.
- Top-Hat Transformation: T_top(f) = f - (f circ B)
- Black-Hat Transformation: T_black(f) = (f bullet B) - f
"""

import cv2
import numpy as np
from scipy import ndimage
from src.morphology.operators import get_structuring_element


def fill_morphological_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill internal voids within binary foreground regions using geodesic hole filling.

    Args:
        binary_mask: Binary array (H, W), uint8 in {0, 255}.

    Returns:
        Filled binary array (H, W), uint8 in {0, 255}.
    """
    bool_mask = binary_mask > 0
    filled_bool = ndimage.binary_fill_holes(bool_mask)
    return (filled_bool * 255).astype(np.uint8)


def apply_tophat(img: np.ndarray, ksize: int = 9, shape: str = "rect") -> np.ndarray:
    """
    Apply White Top-Hat Transform: T_top(f) = f - (f circ B).
    Isolates elements brighter than their surroundings.
    """
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def apply_blackhat(img: np.ndarray, ksize: int = 9, shape: str = "rect") -> np.ndarray:
    """
    Apply Black-Hat Transform: T_black(f) = (f bullet B) - f.
    Isolates elements darker than their surroundings.
    """
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def morphological_reconstruction(
    marker: np.ndarray,
    mask: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Perform classical geodesic dilation reconstruction of marker conditioned on mask:
        D_G^{(1)}(F) = (F (+) B) cap G
    """
    kernel = get_structuring_element("cross", 3)
    current = np.minimum(marker, mask)

    for _ in range(max_iter):
        dilated = cv2.dilate(current, kernel)
        next_iter = np.minimum(dilated, mask)
        if np.array_equal(next_iter, current):
            break
        current = next_iter

    return current
