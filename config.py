"""
Configuration Module for Classical DIP Image Forensics.

Contains centralized thresholds, quality factors, and filter specifications.
All parameters strictly adhere to classical digital image processing standards.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0


@dataclass(frozen=True)
class FrequencyConfig:
    homomorphic_gamma_low: float = 0.5
    homomorphic_gamma_high: float = 2.0
    homomorphic_c: float = 1.0
    homomorphic_d0: float = 30.0
    butterworth_order: int = 2
    epsilon: float = 1e-5


@dataclass(frozen=True)
class CompressionConfig:
    ela_resave_quality: int = 90
    ela_multiplier: float = 15.0
    block_size: int = 8
    variance_flat_thresh: float = 10.0


@dataclass(frozen=True)
class DetectionConfig:
    sift_n_features: int = 1500
    g2nn_ratio: float = 0.65
    min_spatial_dist: float = 20.0
    ransac_reproj_thresh: float = 3.0
    min_inliers_count: int = 5
    splice_weight_ela: float = 0.5
    splice_weight_homomorphic: float = 0.3
    splice_weight_bag: float = 0.2


@dataclass(frozen=True)
class MorphologyConfig:
    opening_kernel_size: int = 3
    closing_kernel_size: int = 7
    min_blob_area: int = 150
