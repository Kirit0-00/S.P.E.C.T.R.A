"""
Deterministic In-Memory Synthetic Forgery Testbed.

Generates reproducible authentic images, copy-move forgeries, and spliced forgeries
with pixel-accurate ground-truth binary masks for automated verification.
"""

import cv2
import numpy as np


def generate_synthetic_base(height: int = 256, width: int = 256) -> np.ndarray:
    """
    Generate a textured synthetic scene with harmonic sinusoidal gratings,
    geometric shapes, and distinct local contrast patterns.
    """
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    # Compound spatial frequencies
    pattern1 = 127.0 + 60.0 * np.sin(x / 8.0) * np.cos(y / 8.0)
    pattern2 = 40.0 * np.sin(np.sqrt((x - 128.0) ** 2 + (y - 128.0) ** 2) / 6.0)
    base = np.clip(pattern1 + pattern2, 0, 255).astype(np.uint8)

    # 3-channel BGR image
    b = base
    g = cv2.circle(base.copy(), (64, 64), 30, (200,), -1)
    r = cv2.rectangle(base.copy(), (160, 160), (220, 220), (230,), -1)
    color_img = cv2.merge([b, g, r])
    return color_img


def create_synthetic_copy_move(
    base_img: np.ndarray,
    source_box: tuple[int, int, int, int] = (40, 40, 60, 60),
    dest_pos: tuple[int, int] = (150, 140),
    rotation_deg: float = 0.0,
    feather_ksize: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a copy-move forgery with geometric transformation and ground-truth mask.

    Args:
        base_img: Source image (H, W, 3), uint8.
        source_box: (x, y, width, height) of source patch.
        dest_pos: (x, y) top-left target location.
        rotation_deg: Rotation angle in degrees applied to cloned patch.
        feather_ksize: Gaussian blur kernel size on boundaries (0 for hard edges).

    Returns:
        Tuple of (forged_image, binary_gt_mask) where mask is 255 at cloned target.
    """
    h, w = base_img.shape[:2]
    sx, sy, sw, sh = source_box
    dx, dy = dest_pos

    forged = base_img.copy()
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    patch = base_img[sy : sy + sh, sx : sx + sw].copy()

    if rotation_deg != 0.0:
        center = (sw // 2, sh // 2)
        m = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        patch = cv2.warpAffine(patch, m, (sw, sh), borderMode=cv2.BORDER_REFLECT)

    patch_mask = np.ones((sh, sw), dtype=np.uint8) * 255

    # Boundary clipping
    target_h = min(sh, h - dy)
    target_w = min(sw, w - dx)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Destination position lies completely outside the image bounds.")

    if feather_ksize > 1:
        feather = cv2.GaussianBlur(patch_mask, (feather_ksize, feather_ksize), 0)
        alpha = (feather[:target_h, :target_w, None] / 255.0).astype(np.float64)
        forged_region = forged[dy : dy + target_h, dx : dx + target_w].astype(np.float64)
        patch_region = patch[:target_h, :target_w].astype(np.float64)
        blended = alpha * patch_region + (1.0 - alpha) * forged_region
        forged[dy : dy + target_h, dx : dx + target_w] = blended.astype(np.uint8)
    else:
        forged[dy : dy + target_h, dx : dx + target_w] = patch[:target_h, :target_w]

    gt_mask[dy : dy + target_h, dx : dx + target_w] = 255
    return forged, gt_mask


def create_synthetic_splice(
    host_img: np.ndarray,
    donor_quality: int = 60,
    splice_box: tuple[int, int, int, int] = (120, 120, 70, 70),
    illumination_boost: float = 1.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesize an image splice with differing JPEG compression and illumination gradient.

    Returns:
        Tuple of (spliced_image, binary_gt_mask).
    """
    h, w = host_img.shape[:2]
    sx, sy, sw, sh = splice_box
    forged = host_img.copy()
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    # Re-encode donor patch with altered JPEG compression quality
    donor_patch = host_img[sy : sy + sh, sx : sx + sw].copy()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), donor_quality]
    _, enc = cv2.imencode(".jpg", donor_patch, encode_param)
    compressed_donor = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Introduce subtle illumination divergence
    donor_float = compressed_donor.astype(np.float64) * illumination_boost
    compressed_donor = np.clip(donor_float, 0, 255).astype(np.uint8)

    # Paste into host
    forged[sy : sy + sh, sx : sx + sw] = compressed_donor
    gt_mask[sy : sy + sh, sx : sx + sw] = 255
    return forged, gt_mask
