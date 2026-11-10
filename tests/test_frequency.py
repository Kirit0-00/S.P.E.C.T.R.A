"""
Unit Tests for Frequency Domain Module and Homomorphic Filtering.
Verifies Fourier transform invariants, filtering operations, and illumination separation.
"""

import numpy as np
import pytest

from src.frequency.fft_core import (
    compute_2d_fft,
    compute_2d_ifft,
    get_magnitude_spectrum,
    get_phase_spectrum,
)
from src.frequency.frequency_filters import (
    compute_distance_matrix,
    create_gaussian_filter,
    create_butterworth_filter,
    create_ideal_filter,
    apply_frequency_filter,
)
from src.frequency.homomorphic import (
    create_homomorphic_filter,
    apply_homomorphic_filter,
)
from tests.synthetic_forger import generate_synthetic_base


def test_fft_roundtrip_invariance():
    img = generate_synthetic_base(128, 128)
    gray = (img[:, :, 0]).astype(np.float64) / 255.0

    f_shifted, orig_shape = compute_2d_fft(gray, pad_optimal=False)
    reconstructed = compute_2d_ifft(f_shifted, target_shape=orig_shape)

    # Invertibility: f == IFFT(FFT(f))
    assert np.allclose(gray, reconstructed, atol=1e-5)


def test_magnitude_and_phase_spectra():
    img = generate_synthetic_base(64, 64)
    f_shifted, _ = compute_2d_fft(img, pad_optimal=True)

    mag = get_magnitude_spectrum(f_shifted)
    phase = get_phase_spectrum(f_shifted)

    assert mag.shape == f_shifted.shape
    assert 0.0 <= mag.max() <= 1.0
    assert -np.pi <= phase.min() and phase.max() <= np.pi


def test_frequency_filters():
    shape = (64, 64)
    d = compute_distance_matrix(shape)
    assert d.shape == shape
    assert np.isclose(d[32, 32], 0.0)

    # GLPF and GHPF
    glpf = create_gaussian_filter(shape, d0=15.0, high_pass=False)
    ghpf = create_gaussian_filter(shape, d0=15.0, high_pass=True)
    assert np.allclose(glpf + ghpf, 1.0)

    # BLPF and BHPF
    blpf = create_butterworth_filter(shape, d0=20.0, order=2, high_pass=False)
    bhpf = create_butterworth_filter(shape, d0=20.0, order=2, high_pass=True)
    assert blpf[32, 32] > 0.99
    assert bhpf[32, 32] < 0.01

    # Ideal filter
    ilpf = create_ideal_filter(shape, d0=10.0, high_pass=False)
    assert ilpf[32, 32] == 1.0


def test_homomorphic_filtering():
    img = generate_synthetic_base(100, 100)
    enhanced, residual = apply_homomorphic_filter(
        img,
        d0=25.0,
        gamma_low=0.4,
        gamma_high=1.8,
    )

    assert enhanced.shape == (100, 100)
    assert residual.shape == (100, 100)
    assert 0.0 <= enhanced.min() and enhanced.max() <= 1.0
    assert 0.0 <= residual.min() and residual.max() <= 1.0
