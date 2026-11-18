"""
Block Artifact Grid (BAG) and 8x8 DCT Boundary Analysis.

Theoretical Foundation (Gonzalez & Woods, Chapter 8.2):
- JPEG standard partitions images into disjoint 8x8 spatial blocks prior to 2D DCT.
- Due to independent block quantization, block boundaries exhibit subtle step discontinuities.
- Boundary Discontinuity Operator:
  D_h(x, y) = |2 * I(x, y) - I(x+1, y) - I(x-1, y)|
  D_v(x, y) = |2 * I(x, y) - I(x, y+1) - I(x, y-1)|
- Forensic Splice Invariant:
  Pasting a spliced patch onto a host image disrupts the continuous 8x8 periodic grid,
  creating localized grid misalignment and boundary phase errors.
"""

import cv2
import numpy as np
from src.spatial.transforms import to_grayscale


def compute_block_artifact_grid(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Extract the Block Artifact Grid (BAG) response across image coordinates.

    Args:
        img: Input array (H, W) or (H, W, 3), uint8.
        block_size: JPEG DCT standard block size (default 8).

    Returns:
        BAG discontinuity heatmap (H, W), float64 normalized to [0.0, 1.0].
    """
    gray = to_grayscale(img).astype(np.float64) / 255.0
    h, w = gray.shape

    # Horizontal second-difference filter: [1, -2, 1]
    kernel_h = np.array([[1.0, -2.0, 1.0]], dtype=np.float64)
    # Vertical second-difference filter: [1; -2; 1]
    kernel_v = np.array([[1.0], [-2.0], [1.0]], dtype=np.float64)

    dh = np.abs(cv2.filter2D(gray, -1, kernel_h, borderType=cv2.BORDER_REFLECT))
    dv = np.abs(cv2.filter2D(gray, -1, kernel_v, borderType=cv2.BORDER_REFLECT))

    bag_response = (dh + dv) / 2.0

    # Smooth within blocks to highlight localized grid disruption
    box_filter = np.ones((block_size, block_size), dtype=np.float64) / (block_size**2)
    smoothed_bag = cv2.filter2D(bag_response, -1, box_filter, borderType=cv2.BORDER_REFLECT)

    max_val = np.max(smoothed_bag)
    if max_val > 0:
        smoothed_bag /= max_val

    return np.clip(smoothed_bag, 0.0, 1.0)


def compute_grid_periodicity(img: np.ndarray, block_size: int = 8) -> dict[str, float]:
    """
    Measure periodicity strength at 8-pixel intervals across horizontal and vertical axes.

    Returns:
        Dictionary with horizontal and vertical grid alignment coherence scores in [0.0, 1.0].
    """
    gray = to_grayscale(img).astype(np.float64)
    h, w = gray.shape

    # Compute row and column differences
    diff_h = np.abs(np.diff(gray, axis=1))  # (H, W-1)
    diff_v = np.abs(np.diff(gray, axis=0))  # (H-1, W)

    # Accumulate differences along axes modulo block_size
    col_profile = np.mean(diff_h, axis=0)  # (W-1,)
    row_profile = np.mean(diff_v, axis=0)  # (H-1,)

    mod_col = np.zeros(block_size, dtype=np.float64)
    for i, val in enumerate(col_profile):
        mod_col[i % block_size] += val

    mod_row = np.zeros(block_size, dtype=np.float64)
    for j, val in enumerate(row_profile):
        mod_row[j % block_size] += val

    # Periodicity peak ratio
    c_peak = (np.max(mod_col) - np.min(mod_col)) / (np.mean(mod_col) + 1e-8)
    r_peak = (np.max(mod_row) - np.min(mod_row)) / (np.mean(mod_row) + 1e-8)

    return {
        "horizontal_grid_score": float(np.clip(c_peak, 0.0, 1.0)),
        "vertical_grid_score": float(np.clip(r_peak, 0.0, 1.0)),
    }
