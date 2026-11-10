"""
Information Entropy and Randomness Mapping.

Theoretical Foundation (Gonzalez & Woods, Chapter 8.1 - Information Theory):
- Shannon Information Entropy:
  H = - sum_{k=0}^{L-1} p(r_k) * log_2(p(r_k))
  where p(r_k) is the probability of intensity level r_k in local window W.
- Forensic Utility:
  Natural camera sensor noise and uncompressed textures exhibit high entropy,
  while flat pasted patches or heavily quantized JPEG regions display reduced entropy.
"""

import numpy as np
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk, square
from src.spatial.transforms import to_grayscale


def compute_local_entropy(
    img: np.ndarray,
    radius: int = 5,
    structuring_element: str = "disk",
) -> np.ndarray:
    """
    Compute local Shannon entropy across neighborhood footprints.

    Args:
        img: Input array (H, W) or (H, W, 3), uint8.
        radius: Structuring element radius / half-width.
        structuring_element: 'disk' or 'square'.

    Returns:
        Local entropy map (H, W), float64 normalized to [0.0, 1.0].
    """
    gray = to_grayscale(img)

    if structuring_element == "square":
        selem = square(2 * radius + 1)
    else:
        selem = disk(radius)

    # skimage rank.entropy returns float64 entropy in bits [0, 8]
    ent = rank_entropy(gray, selem)
    max_entropy = 8.0  # Theoretical max for 8-bit image: log2(256) = 8
    normalized_ent = np.clip(ent / max_entropy, 0.0, 1.0)
    return normalized_ent.astype(np.float64)


def compute_global_entropy(img: np.ndarray) -> float:
    """
    Compute the overall scalar Shannon entropy of an image.

    Returns:
        Entropy value in bits [0.0, 8.0].
    """
    gray = to_grayscale(img)
    counts = np.bincount(gray.ravel(), minlength=256)
    probs = counts / gray.size
    non_zero = probs[probs > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))
