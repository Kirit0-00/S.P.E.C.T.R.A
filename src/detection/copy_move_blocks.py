"""
Block-Based DCT Copy-Move Detection for Flat / Low-Texture Regions.

Theoretical Foundation (Gonzalez & Woods, Chapter 8 - Image Compression):
- JPEG encodes images using 8×8 block DCT. When a region is cloned, its DCT
  coefficient fingerprint is duplicated somewhere else in the image.
- Algorithm:
    1. Divide image into overlapping blocks of size (block_size × block_size).
    2. Compute DCT features for each block (flattened DCT coefficients).
    3. Sort all (block_feature, position) pairs lexicographically.
    4. Find adjacent pairs in sorted order with L2 distance below threshold.
    5. Only accept pairs where spatial displacement > min_dist (not same block).
    6. Project matched block positions onto a binary anomaly heatmap.

This complements keypoint-based detection (copy_move.py) which fails on
flat/textureless regions where SIFT yields no keypoints.
"""

import cv2
import numpy as np
from scipy.fft import dctn
from src.spatial.transforms import to_grayscale


def _extract_block_dct_features(
    gray: np.ndarray,
    block_size: int = 16,
    step: int = 4,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Extract flattened top-left-biased DCT features from overlapping blocks.

    Returns:
        features: (N, block_size*block_size) float32 array of DCT vectors.
        positions: list of (row, col) top-left corner for each block.
    """
    h, w = gray.shape
    features, positions = [], []

    for r in range(0, h - block_size + 1, step):
        for c in range(0, w - block_size + 1, step):
            block = gray[r:r + block_size, c:c + block_size].astype(np.float32)
            # 2D DCT-II (norm='ortho' ensures orthonormal basis)
            dct_block = dctn(block, norm="ortho")
            features.append(dct_block.ravel())
            positions.append((r, c))

    if not features:
        return np.empty((0, block_size * block_size), dtype=np.float32), []

    return np.array(features, dtype=np.float32), positions


def _find_duplicate_blocks(
    features: np.ndarray,
    positions: list[tuple[int, int]],
    threshold: float = 0.02,
    min_dist: int = 16,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Find block pairs whose DCT feature vectors are within L2 `threshold`.
    Uses lexicographic sort + sequential scan for O(N log N) complexity.

    Returns:
        List of ((r1, c1), (r2, c2)) block-position duplicate pairs.
    """
    n = len(features)
    if n < 2:
        return []

    # Normalize each feature vector to unit length (illumination invariant)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norms

    # Lexicographic sort by feature vector content
    sort_idx = np.lexsort(normed.T[::-1])
    sorted_feats = normed[sort_idx]
    sorted_pos = [positions[i] for i in sort_idx]

    pairs = []
    for i in range(n - 1):
        diff = np.linalg.norm(sorted_feats[i] - sorted_feats[i + 1])
        if diff < threshold:
            r1, c1 = sorted_pos[i]
            r2, c2 = sorted_pos[i + 1]
            spatial_dist = np.hypot(r1 - r2, c1 - c2)
            if spatial_dist >= min_dist:
                pairs.append(((r1, c1), (r2, c2)))

    return pairs


def detect_copy_move_blocks(
    img: np.ndarray,
    block_size: int = 16,
    step: int = 4,
    threshold: float = 0.02,
    min_dist: int = 16,
    min_pairs: int = 3,
) -> tuple[np.ndarray, bool]:
    """
    Block-based DCT copy-move detector for flat / textureless regions.

    Args:
        img:        Input image (H, W, 3) or (H, W), uint8.
        block_size: Pixel dimension of each overlapping block (default 16).
        step:       Sliding window step size in pixels (default 4).
        threshold:  L2 distance threshold for declaring two blocks identical.
        min_dist:   Minimum spatial separation (px) between matched block centres.
        min_pairs:  Minimum number of duplicate pairs to declare detection.

    Returns:
        Tuple of:
          - heatmap (H, W) float64 in [0.0, 1.0]: density of duplicate blocks.
          - is_detected bool: True when duplicate pair count >= min_pairs.
    """
    gray = to_grayscale(img)
    h, w = gray.shape
    heatmap = np.zeros((h, w), dtype=np.float64)

    features, positions = _extract_block_dct_features(gray, block_size, step)
    if len(features) < 2:
        return heatmap, False

    pairs = _find_duplicate_blocks(features, positions, threshold, min_dist)

    if len(pairs) < min_pairs:
        return heatmap, False

    # Accumulate block footprints onto heatmap
    for (r1, c1), (r2, c2) in pairs:
        heatmap[r1:r1 + block_size, c1:c1 + block_size] += 1.0
        heatmap[r2:r2 + block_size, c2:c2 + block_size] += 1.0

    # Smooth and normalize
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    max_v = np.max(heatmap)
    if max_v > 0:
        heatmap /= max_v

    return heatmap, True
