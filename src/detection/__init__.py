"""
Forensic Detection Package (Copy-Move & Splice Detection).
"""

from src.detection.copy_move import (
    extract_features_with_variance_gating,
    match_copy_move_features,
    detect_copy_move,
)
from src.detection.splice_detector import detect_splice_anomalies

__all__ = [
    "extract_features_with_variance_gating",
    "match_copy_move_features",
    "detect_copy_move",
    "detect_splice_anomalies",
]
