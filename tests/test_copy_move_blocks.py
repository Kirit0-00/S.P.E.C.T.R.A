"""
Unit tests for Block-based DCT Copy-Move Detection (copy_move_blocks.py).
"""

import numpy as np
import pytest
from src.detection.copy_move_blocks import (
    _extract_block_dct_features,
    _find_duplicate_blocks,
    detect_copy_move_blocks,
)
from tests.synthetic_forger import create_synthetic_copy_move, generate_synthetic_base


def test_extract_block_dct_features_shape():
    """Verify extracted DCT features match expected dimension and step count."""
    gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    block_size = 16
    step = 8
    features, positions = _extract_block_dct_features(gray, block_size=block_size, step=step)

    # Number of blocks along each dimension = (64 - 16) // 8 + 1 = 7
    expected_count = 7 * 7
    assert len(features) == expected_count
    assert len(positions) == expected_count
    assert features.shape[1] == block_size * block_size


def test_detect_copy_move_blocks_synthetic():
    """Verify block detector catches a synthetic copy-move forgery."""
    base = generate_synthetic_base(256, 256)
    forged, mask = create_synthetic_copy_move(
        base,
        source_box=(30, 30, 80, 80),
        dest_pos=(140, 140),
    )
    heatmap, is_detected = detect_copy_move_blocks(
        forged,
        block_size=16,
        step=4,
        threshold=0.05,
        min_dist=20,
        min_pairs=2,
    )
    assert heatmap.shape == (256, 256)
    assert heatmap.dtype == np.float64
    assert is_detected is True
    assert np.max(heatmap) > 0.0


def test_detect_copy_move_blocks_authentic():
    """Verify detector returns low/no detection on authentic image with high min_pairs."""
    # Authentic random noise image has no identical blocks
    np.random.seed(42)
    clean = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    heatmap, is_detected = detect_copy_move_blocks(
        clean,
        block_size=16,
        step=8,
        threshold=0.001,
        min_dist=20,
        min_pairs=5,
    )
    assert heatmap.shape == (128, 128)
    assert is_detected is False
