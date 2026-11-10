"""
Error Level Analysis (ELA) for JPEG Compression Discrepancies.

Theoretical Foundation (Gonzalez & Woods, Chapter 8 - Image Compression):
- JPEG Compression Loss: Involves 8x8 Block DCT, Quantization with matrix Q,
  and Huffman coding.
- Error Level Analysis: Re-saving an image at a known quality factor Q_ref
  produces an error residual E(x, y) = |f(x, y) - f_{Q}(x, y)|.
- Compression Invariant: Authentic regions previously compressed at quality >= Q_ref
  reach an equilibrium state with minimal residual error. Spliced regions from uncompressed
  or differently compressed sources display elevated error variance.
"""

import cv2
import numpy as np
from src.spatial.transforms import normalize_to_uint8


def compute_ela(
    img: np.ndarray,
    quality: int = 90,
    multiplier: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Error Level Analysis (ELA) residual matrix.

    Args:
        img: Input image (H, W, 3) or (H, W), uint8.
        quality: JPEG re-compression quality factor in [1, 100].
        multiplier: Scaling factor for visualizing subtle residuals.

    Returns:
        Tuple of (ela_visual uint8 in [0, 255], ela_raw float64 in [0.0, 1.0]).
    """
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    # Encode to in-memory JPEG stream
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
    if not success:
        raise ValueError("JPEG encoding failed during ELA processing.")

    # Decode back to spatial domain
    resaved = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # Compute channel-wise absolute difference
    diff = np.abs(img_bgr.astype(np.float64) - resaved.astype(np.float64))

    # Normalized float anomaly heatmap (average across color channels)
    ela_raw = np.mean(diff, axis=2) / 255.0

    # Scaled uint8 visualization
    scaled_diff = diff * multiplier
    ela_visual = np.clip(scaled_diff, 0, 255).astype(np.uint8)

    return ela_visual, ela_raw


def compute_ela_variance(ela_raw: np.ndarray, ksize: int = 9) -> np.ndarray:
    """
    Compute local variance of ELA residuals to highlight non-uniform compression regions.

    Args:
        ela_raw: Grayscale normalized ELA residual map (H, W), float64.
        ksize: Local statistical window dimension.

    Returns:
        Variance map (H, W), float64 normalized to [0.0, 1.0].
    """
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    mean = cv2.filter2D(ela_raw, -1, kernel, borderType=cv2.BORDER_REFLECT)
    sqr_mean = cv2.filter2D(ela_raw**2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(sqr_mean - mean**2, 0.0)

    max_var = np.max(var)
    if max_var > 0:
        var /= max_var
    return var
