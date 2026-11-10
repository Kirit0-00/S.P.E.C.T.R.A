"""
Unit Tests for Spatial Domain Module.
Verifies data contracts, shapes, numerical boundaries, and DIP compliance.
"""

import numpy as np
import pytest

from src.spatial.transforms import (
    to_grayscale,
    to_float64,
    normalize_to_uint8,
    log_transform,
    gamma_transform,
)
from src.spatial.equalization import (
    global_histogram_equalization,
    apply_clahe,
    compute_normalized_histogram,
)
from src.spatial.linear_filters import (
    apply_gaussian_blur,
    apply_bilateral_filter,
    compute_sobel_gradients,
    apply_laplacian,
    apply_unsharp_mask,
    compute_local_variance,
)
from tests.synthetic_forger import generate_synthetic_base


def test_spatial_transforms():
    img = generate_synthetic_base(128, 128)
    assert img.shape == (128, 128, 3)

    gray = to_grayscale(img)
    assert gray.shape == (128, 128)
    assert gray.dtype == np.uint8

    f64 = to_float64(img)
    assert f64.dtype == np.float64
    assert 0.0 <= f64.min() and f64.max() <= 1.0

    u8 = normalize_to_uint8(f64)
    assert u8.dtype == np.uint8

    log_img = log_transform(gray)
    assert log_img.dtype == np.float64
    assert log_img.shape == (128, 128)

    gamma_img = gamma_transform(gray, gamma=0.5)
    assert gamma_img.dtype == np.float64
    assert 0.0 <= gamma_img.min() and gamma_img.max() <= 1.0


def test_equalization():
    img = generate_synthetic_base(128, 128)
    gray = to_grayscale(img)

    eq_gray = global_histogram_equalization(gray)
    assert eq_gray.shape == (128, 128)
    assert eq_gray.dtype == np.uint8

    eq_color = global_histogram_equalization(img)
    assert eq_color.shape == (128, 128, 3)
    assert eq_color.dtype == np.uint8

    clahe_out = apply_clahe(gray, clip_limit=2.0)
    assert clahe_out.shape == (128, 128)
    assert clahe_out.dtype == np.uint8

    hist = compute_normalized_histogram(gray)
    assert hist.shape == (256,)
    assert np.isclose(np.sum(hist), 1.0)


def test_linear_filters():
    img = generate_synthetic_base(128, 128)
    gray = to_grayscale(img)

    # Gaussian blur
    blurred = apply_gaussian_blur(gray, ksize=5, sigma=1.0)
    assert blurred.shape == (128, 128)

    # Bilateral filter
    bilateral = apply_bilateral_filter(img, d=5, sigma_color=50, sigma_space=50)
    assert bilateral.shape == (128, 128, 3)

    # Sobel gradients
    gx, gy, mag = compute_sobel_gradients(gray)
    assert gx.shape == (128, 128)
    assert gy.shape == (128, 128)
    assert mag.shape == (128, 128)
    assert 0.0 <= mag.max() <= 1.0

    # Laplacian
    lap = apply_laplacian(gray)
    assert lap.shape == (128, 128)
    assert 0.0 <= lap.max() <= 1.0

    # Unsharp mask
    sharpened = apply_unsharp_mask(img, ksize=5, sigma=1.0, k=1.5)
    assert sharpened.shape == (128, 128, 3)
    assert sharpened.dtype == np.uint8

    # Local variance
    var_map = compute_local_variance(gray, ksize=7)
    assert var_map.shape == (128, 128)
    assert np.all(var_map >= 0.0)
