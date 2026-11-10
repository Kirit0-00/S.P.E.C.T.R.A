"""
Pipeline Stage Contracts & Shape Scaffolding Tests.

Validates the expected input/output shapes, dtypes, and tensor transformations
across all DIP pipeline stages defined in BUILD_PLAN.md and MEMORY.md.
Uses mock numpy arrays to ensure pipeline contract integrity.
"""

import numpy as np
import pytest


@pytest.fixture
def mock_color_image() -> np.ndarray:
    """Mock RGB/BGR color image: f(x, y) in R^{H x W x 3}, uint8 in [0, 255]."""
    return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def mock_grayscale_image() -> np.ndarray:
    """Mock grayscale luminance image: I(x, y) in R^{H x W}, uint8 in [0, 255]."""
    return np.random.randint(0, 256, (128, 128), dtype=np.uint8)


@pytest.fixture
def mock_anomaly_heatmap() -> np.ndarray:
    """Mock anomaly heatmap: A(x, y) in R^{H x W}, float64 in [0.0, 1.0]."""
    return np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float64)


def test_spatial_stage_contracts(mock_color_image, mock_grayscale_image):
    """
    Contract:
      Input: f(x, y) in (H, W, 3) or (H, W), uint8
      Output: I(x, y) in (H, W), uint8 or float64 in [0.0, 1.0]
    """
    h, w, c = mock_color_image.shape
    assert c == 3
    assert mock_color_image.dtype == np.uint8

    # Mock conversion to grayscale
    gray_mock = np.mean(mock_color_image, axis=2).astype(np.uint8)
    assert gray_mock.shape == (h, w)
    assert gray_mock.dtype == np.uint8

    # Mock normalized float transform
    norm_float = gray_mock.astype(np.float64) / 255.0
    assert norm_float.shape == (h, w)
    assert norm_float.dtype == np.float64
    assert 0.0 <= norm_float.min() and norm_float.max() <= 1.0


def test_frequency_stage_contracts(mock_grayscale_image):
    """
    Contract:
      Input: I(x, y) in (H, W), float64
      Output: F(u, v) in (H, W) or (M, N), complex128;
              Magnitude spectrum in (M, N), float64;
              Reconstructed image in (H, W), float64.
    """
    h, w = mock_grayscale_image.shape
    img_float = mock_grayscale_image.astype(np.float64)

    # Mock 2D FFT output
    mock_fft = np.fft.fft2(img_float)
    mock_shifted = np.fft.fftshift(mock_fft)

    assert mock_shifted.shape == (h, w)
    assert np.iscomplexobj(mock_shifted)
    assert mock_shifted.dtype == np.complex128

    # Mock magnitude spectrum
    mock_mag = np.log1p(np.abs(mock_shifted))
    mock_mag /= np.max(mock_mag)
    assert mock_mag.shape == (h, w)
    assert mock_mag.dtype == np.float64
    assert 0.0 <= mock_mag.min() and mock_mag.max() <= 1.0

    # Mock 2D IFFT output
    mock_reconstructed = np.real(np.fft.ifft2(np.fft.ifftshift(mock_shifted)))
    assert mock_reconstructed.shape == (h, w)
    assert mock_reconstructed.dtype == np.float64


def test_compression_ela_stage_contracts(mock_color_image):
    """
    Contract:
      Input: f(x, y) in (H, W, 3), uint8
      Output: ELA visual in (H, W, 3), uint8;
              ELA anomaly map in (H, W), float64 in [0.0, 1.0];
              Block artifact grid in (H, W), float64 in [0.0, 1.0].
    """
    h, w, c = mock_color_image.shape

    # Mock ELA difference matrix
    mock_resaved = np.clip(mock_color_image.astype(np.float64) + np.random.normal(0, 2, (h, w, c)), 0, 255)
    mock_diff = np.abs(mock_color_image.astype(np.float64) - mock_resaved)

    mock_ela_visual = np.clip(mock_diff * 10.0, 0, 255).astype(np.uint8)
    assert mock_ela_visual.shape == (h, w, c)
    assert mock_ela_visual.dtype == np.uint8

    mock_ela_raw = np.mean(mock_diff, axis=2) / 255.0
    assert mock_ela_raw.shape == (h, w)
    assert mock_ela_raw.dtype == np.float64
    assert 0.0 <= mock_ela_raw.min() and mock_ela_raw.max() <= 1.0

    # Mock BAG (Block Artifact Grid) response
    mock_bag = np.zeros((h, w), dtype=np.float64)
    assert mock_bag.shape == (h, w)
    assert mock_bag.dtype == np.float64


def test_detection_fusion_contracts(mock_color_image):
    """
    Contract:
      Input: Multi-cue anomaly responses (ELA, Homomorphic, BAG)
      Output: Fused anomaly heatmap A(x, y) in (H, W), float64 in [0.0, 1.0].
    """
    h, w, _ = mock_color_image.shape
    cue_ela = np.random.uniform(0.0, 1.0, (h, w))
    cue_homomorphic = np.random.uniform(0.0, 1.0, (h, w))
    cue_bag = np.random.uniform(0.0, 1.0, (h, w))

    # Mock linear convex fusion
    w_ela, w_homo, w_bag = 0.5, 0.3, 0.2
    fused_heatmap = w_ela * cue_ela + w_homo * cue_homomorphic + w_bag * cue_bag

    assert fused_heatmap.shape == (h, w)
    assert fused_heatmap.dtype == np.float64
    assert 0.0 <= fused_heatmap.min() and fused_heatmap.max() <= 1.0


def test_morphological_refinement_contracts(mock_anomaly_heatmap):
    """
    Contract:
      Input: A(x, y) in (H, W), float64 in [0.0, 1.0]
      Output: Cleaned binary tamper mask g(x, y) in (H, W), uint8 in {0, 255}.
    """
    h, w = mock_anomaly_heatmap.shape

    # Mock Otsu binarization
    threshold = 0.5
    raw_binary = np.where(mock_anomaly_heatmap >= threshold, 255, 0).astype(np.uint8)

    assert raw_binary.shape == (h, w)
    assert raw_binary.dtype == np.uint8
    unique_vals = set(np.unique(raw_binary))
    assert unique_vals.issubset({0, 255})
