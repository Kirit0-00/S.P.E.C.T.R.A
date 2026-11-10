"""
Unit Tests for Copy-Move Detection Module.
Verifies keypoint feature extraction, variance gating, g2NN matching, and RANSAC verification.
"""

import numpy as np
import pytest

from src.detection.copy_move import (
    extract_features_with_variance_gating,
    match_copy_move_features,
    detect_copy_move,
)
from tests.synthetic_forger import generate_synthetic_base, create_synthetic_copy_move


def test_feature_extraction_and_variance_gating():
    img = generate_synthetic_base(128, 128)
    kps, descs = extract_features_with_variance_gating(img, n_features=500, var_thresh=5.0)

    # Should detect keypoints in non-flat regions
    assert len(kps) > 0
    assert len(descs) == len(kps)
    assert descs.ndim == 2


def test_copy_move_detection_on_authentic():
    img = generate_synthetic_base(128, 128)
    heatmap, inliers, is_detected = detect_copy_move(
        img,
        n_features=500,
        ratio_thresh=0.6,
        min_spatial_dist=30.0,
        min_inliers=6,
    )

    assert heatmap.shape == (128, 128)
    assert heatmap.dtype == np.float64
    # An authentic image should not trigger a verified geometric copy-move cluster
    assert not is_detected
    assert len(inliers) < 6


def test_copy_move_detection_on_forgery():
    base = generate_synthetic_base(256, 256)
    # Synthesize copy-move cloning a textured box to another location
    forged, gt_mask = create_synthetic_copy_move(
        base,
        source_box=(30, 30, 80, 80),
        dest_pos=(140, 140),
        rotation_deg=0.0,
    )

    heatmap, inliers, is_detected = detect_copy_move(
        forged,
        n_features=1200,
        var_thresh=5.0,
        ratio_thresh=0.75,
        min_spatial_dist=25.0,
        min_inliers=4,
    )

    assert heatmap.shape == (256, 256)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert is_detected
    assert len(inliers) >= 4
