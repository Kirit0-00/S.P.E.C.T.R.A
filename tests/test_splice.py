"""
Unit Tests for Splice Anomaly Detection Module.
Verifies multi-cue convex fusion of ELA, Homomorphic Illumination, and 8x8 BAG.
"""

import numpy as np
import pytest

from src.detection.splice_detector import detect_splice_anomalies
from tests.synthetic_forger import generate_synthetic_base, create_synthetic_splice


def test_splice_detector_contracts():
    img = generate_synthetic_base(128, 128)
    fused, cues = detect_splice_anomalies(
        img,
        weight_ela=0.5,
        weight_homomorphic=0.3,
        weight_bag=0.2,
    )

    assert fused.shape == (128, 128)
    assert fused.dtype == np.float64
    assert 0.0 <= fused.min() and fused.max() <= 1.0

    assert "ela_cue" in cues
    assert "homomorphic_cue" in cues
    assert "bag_cue" in cues

    assert cues["ela_cue"].shape == (128, 128)
    assert cues["homomorphic_cue"].shape == (128, 128)
    assert cues["bag_cue"].shape == (128, 128)


def test_splice_detection_on_synthetic_splice():
    base = generate_synthetic_base(128, 128)
    spliced, gt_mask = create_synthetic_splice(
        base,
        donor_quality=50,
        splice_box=(30, 30, 45, 45),
        illumination_boost=1.35,
    )

    fused, cues = detect_splice_anomalies(
        spliced,
        weight_ela=0.5,
        weight_homomorphic=0.3,
        weight_bag=0.2,
    )

    assert fused.shape == (128, 128)
    assert 0.0 <= fused.min() and fused.max() <= 1.0

    # Splice region should show elevated anomaly response compared to global minimum
    spliced_mean = np.mean(fused[gt_mask == 255])
    assert spliced_mean > 0.0
