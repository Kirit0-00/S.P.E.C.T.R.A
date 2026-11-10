"""
Morphological Processing & Mask Refinement Package.
"""

from src.morphology.operators import (
    get_structuring_element,
    apply_erosion,
    apply_dilation,
    apply_opening,
    apply_closing,
    apply_morphological_gradient,
)
from src.morphology.reconstruction import (
    fill_morphological_holes,
    apply_tophat,
    apply_blackhat,
    morphological_reconstruction,
)
from src.morphology.segmentation import (
    apply_otsu_threshold,
    filter_connected_components,
    extract_morphological_boundary,
    compute_skeleton,
    refine_tamper_mask,
)

__all__ = [
    "get_structuring_element",
    "apply_erosion",
    "apply_dilation",
    "apply_opening",
    "apply_closing",
    "apply_morphological_gradient",
    "fill_morphological_holes",
    "apply_tophat",
    "apply_blackhat",
    "morphological_reconstruction",
    "apply_otsu_threshold",
    "filter_connected_components",
    "extract_morphological_boundary",
    "compute_skeleton",
    "refine_tamper_mask",
]
