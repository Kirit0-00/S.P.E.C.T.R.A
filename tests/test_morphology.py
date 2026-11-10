"""
Unit Tests for Morphological Post-Processing Module.
Verifies erosion, dilation, opening, closing, hole filling, CCA, and mask refinement.
"""

import numpy as np
import pytest

from src.morphology.operators import (
    get_structuring_element,
    apply_erosion,
    apply_dilation,
    apply_opening,
    apply_closing,
    apply_morphological_gradient,
)
from src.morphology.reconstruction import fill_morphological_holes, apply_tophat, apply_blackhat
from src.morphology.segmentation import (
    apply_otsu_threshold,
    filter_connected_components,
    extract_morphological_boundary,
    compute_skeleton,
    refine_tamper_mask,
)


def test_morphological_operators():
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[20:44, 20:44] = 255  # 24x24 square

    eroded = apply_erosion(binary, ksize=3, shape="rect")
    dilated = apply_dilation(binary, ksize=3, shape="rect")

    # Erosion shrinks area; Dilation expands area
    assert np.sum(eroded) < np.sum(binary)
    assert np.sum(dilated) > np.sum(binary)

    opened = apply_opening(binary, ksize=3)
    closed = apply_closing(binary, ksize=3)
    assert opened.shape == (64, 64)
    assert closed.shape == (64, 64)

    grad = apply_morphological_gradient(binary, ksize=3)
    assert grad.shape == (64, 64)
    assert np.any(grad > 0)


def test_hole_filling():
    # Construct a ring with a hole in the center
    ring = np.zeros((64, 64), dtype=np.uint8)
    ring[15:45, 15:45] = 255
    ring[25:35, 25:35] = 0  # The hole

    filled = fill_morphological_holes(ring)
    # The center void should now be solid 255
    assert np.all(filled[25:35, 25:35] == 255)


def test_connected_component_filtering():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Small noise blob (area 9)
    mask[10:13, 10:13] = 255
    # Large genuine blob (area 400)
    mask[40:60, 40:60] = 255

    filtered = filter_connected_components(mask, min_area=50)

    # Small noise blob must be eliminated (0)
    assert np.all(filtered[10:13, 10:13] == 0)
    # Large blob must be preserved (255)
    assert np.all(filtered[40:60, 40:60] == 255)


def test_refine_tamper_mask_pipeline():
    # Create synthetic float anomaly heatmap
    heatmap = np.zeros((100, 100), dtype=np.float64)
    heatmap[30:70, 30:70] = 0.9  # Tamper peak
    heatmap[5:8, 5:8] = 0.85     # Small noise peak

    clean_mask, boundary = refine_tamper_mask(
        heatmap,
        min_area=100,
        opening_ksize=3,
        closing_ksize=5,
    )

    assert clean_mask.shape == (100, 100)
    assert boundary.shape == (100, 100)
    assert set(np.unique(clean_mask)).issubset({0, 255})
    assert set(np.unique(boundary)).issubset({0, 255})

    # Noise peak filtered out
    assert np.all(clean_mask[5:8, 5:8] == 0)
    # Primary region retained
    assert np.any(clean_mask[30:70, 30:70] == 255)
