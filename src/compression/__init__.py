"""
Compression & Information Redundancy Analysis Package.
"""

from src.compression.ela import compute_ela, compute_ela_variance
from src.compression.block_grid import compute_block_artifact_grid, compute_grid_periodicity
from src.compression.entropy import compute_local_entropy, compute_global_entropy

__all__ = [
    "compute_ela",
    "compute_ela_variance",
    "compute_block_artifact_grid",
    "compute_grid_periodicity",
    "compute_local_entropy",
    "compute_global_entropy",
]
