"""Reorganization and biomass estimation module.

The final stage of the BioViT3R pipeline takes a reconstructed 3D point cloud
of a plant and produces an estimate of its above-ground biomass. This
module defines the :class:`BiomassEstimator` class, which encapsulates
species-specific allometric equations and provides helper methods for
computing geometric quantities such as volume and surface area. The
module uses ``scipy`` for computing convex hulls; if not available, it
falls back to a simple bounding box approximation.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from scipy.spatial import ConvexHull
except ImportError:  # pragma: no cover
    ConvexHull = None  # type: ignore


class BiomassEstimator:
    """Estimate plant biomass from geometric measurements.

    The estimator is initialised with a dictionary mapping species
    identifiers (integers) to allometric parameters. Each parameter set
    should include the keys ``a`` and ``b`` corresponding to the power law
    coefficients in the relationship::

        biomass = a * volume ** b

    Additional factors such as density may also be stored. If a species
    identifier is not present in the dictionary, a default set of
    parameters will be used.

    Parameters
    ----------
    allometric_params: Dict[int, Dict[str, float]]
        Dictionary of allometric parameters per species. Each species ID
        should map to a dictionary containing ``a`` and ``b``. Optionally
        ``density`` and other factors can be provided.
    default_params: Optional[Dict[str, float]]
        Default allometric parameters used when a species is not found in
        ``allometric_params``. Should contain at least ``a`` and ``b``.
    """

    def __init__(
        self,
        allometric_params: Dict[int, Dict[str, float]],
        default_params: Optional[Dict[str, float]] = None,
    ) -> None:
        if default_params is None:
            default_params = {"a": 1.0, "b": 1.0}
        self.allometric_params = allometric_params
        self.default_params = default_params

    def estimate_biomass(self, points: np.ndarray, species_id: int) -> float:
        """Estimate biomass from a point cloud for a given species.

        Args:
            points: ``(N, 3)`` array of 3D points representing the plant.
            species_id: Integer identifier for the plant species.

        Returns:
            Estimated biomass (in arbitrary units).
        """
        vol = self.compute_volume(points)
        params = self.allometric_params.get(species_id, self.default_params)
        a = params.get("a", self.default_params["a"])
        b = params.get("b", self.default_params["b"])
        density = params.get("density", 1.0)
        biomass = a * (vol ** b) * density
        return float(biomass)

    @staticmethod
    def compute_volume(points: np.ndarray) -> float:
        """Compute volume of a point cloud via its convex hull.

        If ``scipy`` is available, uses ``ConvexHull`` to compute the volume
        of the hull. If not, falls back to computing the volume of the
        axis-aligned bounding box enclosing the points.

        Args:
            points: ``(N, 3)`` array of 3D points.

        Returns:
            Estimated volume of the hull (or bounding box).
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an array of shape (N, 3)")
        if ConvexHull is not None and points.shape[0] >= 4:
            try:
                hull = ConvexHull(points)
                return float(hull.volume)
            except Exception:
                # if ConvexHull fails (e.g. co-linear points), fall back
                pass
        # Fallback: bounding box volume
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return float(np.prod(maxs - mins))