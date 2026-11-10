"""
Segmentation, Connected Component Analysis, and Tamper Mask Refinement.

Theoretical Foundation (Gonzalez & Woods, Chapters 9.5 & 10.3):
- Otsu's Global Optimal Thresholding: Maximizes between-class variance sigma_B^2(t).
- Connected Component Analysis (CCA): 8-connected labeling and area filtering to eliminate
  spurious false-positive noise blobs.
- Morphological Boundary Extraction: beta(A) = A - (A (-) B_3x3).
- Skeletonization / Thinning: Iterative morphological thinning to extract 1-pixel medials.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from src.morphology.operators import apply_closing, apply_erosion, apply_opening
from src.morphology.reconstruction import fill_morphological_holes


def apply_otsu_threshold(heatmap: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Binarize normalized float anomaly heatmap [0.0, 1.0] using Otsu's optimal method.

    Returns:
        Tuple of (binary_mask uint8 in {0, 255}, optimal_threshold float in [0.0, 1.0]).
    """
    heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    thresh_val, binary = cv2.threshold(heatmap_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    norm_thresh = float(thresh_val / 255.0)
    return binary, norm_thresh


def filter_connected_components(binary_mask: np.ndarray, min_area: int = 150) -> np.ndarray:
    """
    Filter out isolated binary blobs with area smaller than min_area pixels.

    Args:
        binary_mask: Binary array (H, W), uint8 in {0, 255}.
        min_area: Minimum pixel count threshold for genuine tamper clusters.

    Returns:
        Cleaned binary mask (H, W), uint8 in {0, 255}.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def extract_morphological_boundary(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract 1-pixel outer morphological boundary: beta(A) = A - (A (-) B_3x3).
    """
    eroded = apply_erosion(binary_mask, ksize=3, shape="cross")
    boundary = cv2.subtract(binary_mask, eroded)
    return boundary


def compute_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute morphological skeleton using medial axis thinning.
    """
    bool_mask = binary_mask > 0
    skel_bool = skeletonize(bool_mask)
    return (skel_bool * 255).astype(np.uint8)


def refine_tamper_mask(
    anomaly_heatmap: np.ndarray,
    min_area: int = 150,
    opening_ksize: int = 3,
    closing_ksize: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete morphological post-processing pipeline:
    1. Otsu Optimal Thresholding
    2. Morphological Opening (salt-noise suppression)
    3. Morphological Closing (bridging fractured tamper components)
    4. Geodesic Hole Filling (solidifying internal voids)
    5. Connected Component Area Gating
    6. Boundary Extraction

    Returns:
        Tuple of (clean_binary_mask uint8 {0, 255}, boundary_mask uint8 {0, 255}).
    """
    # 1. Otsu Thresholding
    raw_binary, _ = apply_otsu_threshold(anomaly_heatmap)

    # 2. Opening to discard tiny false-positive peaks
    opened = apply_opening(raw_binary, ksize=opening_ksize, shape="ellipse")

    # 3. Closing to fuse contiguous segments
    closed = apply_closing(opened, ksize=closing_ksize, shape="ellipse")

    # 4. Geodesic Hole Filling
    filled = fill_morphological_holes(closed)

    # 5. Connected Component Area Filtering
    filtered = filter_connected_components(filled, min_area=min_area)

    # 6. Morphological Boundary Extraction
    boundary = extract_morphological_boundary(filtered)

    return filtered, boundary
