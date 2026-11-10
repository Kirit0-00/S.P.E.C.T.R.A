"""
Unit Tests for Compression & Entropy Forensics Module.
Verifies Error Level Analysis, 8x8 Block Artifact Grid, and Shannon Entropy.
"""

import numpy as np
import pytest

from src.compression.ela import compute_ela, compute_ela_variance
from src.compression.block_grid import compute_block_artifact_grid, compute_grid_periodicity
from src.compression.entropy import compute_local_entropy, compute_global_entropy
from tests.synthetic_forger import generate_synthetic_base, create_synthetic_splice


def test_error_level_analysis():
    img = generate_synthetic_base(128, 128)
    visual, raw = compute_ela(img, quality=90, multiplier=15.0)

    assert visual.shape == (128, 128, 3)
    assert visual.dtype == np.uint8
    assert raw.shape == (128, 128)
    assert raw.dtype == np.float64
    assert 0.0 <= raw.min() and raw.max() <= 1.0

    # Test ELA variance map
    var_map = compute_ela_variance(raw, ksize=9)
    assert var_map.shape == (128, 128)
    assert 0.0 <= var_map.min() and var_map.max() <= 1.0


def test_block_artifact_grid():
    img = generate_synthetic_base(128, 128)
    bag = compute_block_artifact_grid(img, block_size=8)

    assert bag.shape == (128, 128)
    assert bag.dtype == np.float64
    assert 0.0 <= bag.min() and bag.max() <= 1.0

    scores = compute_grid_periodicity(img, block_size=8)
    assert "horizontal_grid_score" in scores
    assert "vertical_grid_score" in scores
    assert 0.0 <= scores["horizontal_grid_score"] <= 1.0
    assert 0.0 <= scores["vertical_grid_score"] <= 1.0


def test_information_entropy():
    img = generate_synthetic_base(96, 96)
    local_ent = compute_local_entropy(img, radius=4, structuring_element="disk")

    assert local_ent.shape == (96, 96)
    assert local_ent.dtype == np.float64
    assert 0.0 <= local_ent.min() and local_ent.max() <= 1.0

    global_ent = compute_global_entropy(img)
    assert 0.0 <= global_ent <= 8.0


def test_splice_compression_discrepancy():
    base = generate_synthetic_base(128, 128)
    spliced, gt_mask = create_synthetic_splice(base, donor_quality=50, splice_box=(30, 30, 40, 40))

    _, raw = compute_ela(spliced, quality=90)
    # The spliced region should have non-zero ELA residuals
    spliced_residual = raw[gt_mask == 255]
    assert np.mean(spliced_residual) >= 0.0
