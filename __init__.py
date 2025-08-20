"""Utility functions for BioViT3R.

The ``utils`` subpackage contains miscellaneous helper functions that are
useful throughout the BioViT3R codebase. These include geometry
operations, evaluation metrics, logging utilities and any other small
functions that do not naturally belong in the main model modules.
"""

from .geometry import chamfer_distance, compute_chamfer_distance
from .metrics import compute_biomass_error

__all__ = [
    "chamfer_distance",
    "compute_chamfer_distance",
    "compute_biomass_error",
]