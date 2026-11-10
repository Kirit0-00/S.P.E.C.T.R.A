"""
Multi-Cue Splice Anomaly Detector.

Theoretical Foundation:
Combines three orthogonal forensic cues derived from classical DIP domains:
1. Compression Domain: Error Level Analysis (ELA) residual variance.
2. Frequency Domain: Homomorphic illumination field divergence Delta_illum(x, y).
3. Block Domain: JPEG 8x8 Block Artifact Grid (BAG) boundary discontinuities.

Fusion Model:
    A_splice(x, y) = w_ela * E_norm(x, y) + w_homo * Delta_illum(x, y) + w_bag * B(x, y)
"""

import cv2
import numpy as np
from src.compression.block_grid import compute_block_artifact_grid
from src.compression.ela import compute_ela
from src.frequency.homomorphic import apply_homomorphic_filter
from src.spatial.transforms import to_grayscale


def detect_splice_anomalies(
    img: np.ndarray,
    weight_ela: float = 0.5,
    weight_homomorphic: float = 0.3,
    weight_bag: float = 0.2,
    ela_quality: int = 90,
    homomorphic_d0: float = 30.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute multi-cue composite splice anomaly heatmap.

    Args:
        img: Input image (H, W, 3) or (H, W), uint8.
        weight_ela: Relative contribution of ELA residual map.
        weight_homomorphic: Relative contribution of illumination discrepancy.
        weight_bag: Relative contribution of block artifact grid discontinuity.
        ela_quality: JPEG quality for ELA re-compression.
        homomorphic_d0: Frequency cutoff for homomorphic filtering.

    Returns:
        Tuple of:
          - fused_heatmap: (H, W), float64 normalized to [0.0, 1.0]
          - intermediate_cues: Dictionary containing individual normalized cue maps.
    """
    # 1. Error Level Analysis (Compression Domain)
    _, ela_raw = compute_ela(img, quality=ela_quality)
    # Enhance local contrast of ELA
    ela_norm = cv2.GaussianBlur(ela_raw, (5, 5), 0)
    max_ela = np.max(ela_norm)
    if max_ela > 0:
        ela_norm /= max_ela

    # 2. Homomorphic Illumination Field Discrepancy (Frequency Domain)
    _, homo_residual = apply_homomorphic_filter(img, d0=homomorphic_d0)
    homo_norm = cv2.GaussianBlur(homo_residual, (7, 7), 0)
    max_homo = np.max(homo_norm)
    if max_homo > 0:
        homo_norm /= max_homo

    # 3. 8x8 Block Artifact Grid (BAG) Discontinuities
    bag_map = compute_block_artifact_grid(img, block_size=8)

    # 4. Convex Linear Combination
    total_weight = weight_ela + weight_homomorphic + weight_bag
    w_e = weight_ela / total_weight
    w_h = weight_homomorphic / total_weight
    w_b = weight_bag / total_weight

    fused = (w_e * ela_norm) + (w_h * homo_norm) + (w_b * bag_map)

    # Dynamic contrast normalization
    fused_norm = np.clip(fused, 0.0, 1.0)
    max_f = np.max(fused_norm)
    min_f = np.min(fused_norm)
    if not np.isclose(max_f, min_f):
        fused_norm = (fused_norm - min_f) / (max_f - min_f)

    intermediate_cues = {
        "ela_cue": ela_norm,
        "homomorphic_cue": homo_norm,
        "bag_cue": bag_map,
    }

    return fused_norm, intermediate_cues
