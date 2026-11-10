"""
Spatial Domain Preprocessing and Filtering Package.
"""

from src.spatial.transforms import (
    to_grayscale,
    to_float64,
    normalize_to_uint8,
    log_transform,
    gamma_transform,
)
from src.spatial.equalization import (
    global_histogram_equalization,
    apply_clahe,
    compute_normalized_histogram,
)
from src.spatial.linear_filters import (
    apply_gaussian_blur,
    apply_bilateral_filter,
    compute_sobel_gradients,
    apply_laplacian,
    apply_unsharp_mask,
    compute_local_variance,
)

__all__ = [
    "to_grayscale",
    "to_float64",
    "normalize_to_uint8",
    "log_transform",
    "gamma_transform",
    "global_histogram_equalization",
    "apply_clahe",
    "compute_normalized_histogram",
    "apply_gaussian_blur",
    "apply_bilateral_filter",
    "compute_sobel_gradients",
    "apply_laplacian",
    "apply_unsharp_mask",
    "compute_local_variance",
]
