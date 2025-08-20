"""Geometry helper functions.

This module contains functions related to geometric processing of point
clouds. The primary utility provided here is the Chamfer distance
computation, which measures how close two point sets are to each other.
"""
from __future__ import annotations

import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None  # type: ignore


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute the symmetric Chamfer distance between two point clouds.

    The Chamfer distance is defined as the sum of the average squared
    distances from each point in one set to its nearest neighbour in the
    other set, and vice versa. This function returns the average of
    distances rather than the sum, providing a scale-invariant metric.

    Args:
        points_a: ``(N, 3)`` array of points.
        points_b: ``(M, 3)`` array of points.

    Returns:
        The symmetric Chamfer distance as a float.
    """
    if points_a.ndim != 2 or points_b.ndim != 2 or points_a.shape[1] != 3 or points_b.shape[1] != 3:
        raise ValueError("inputs must be arrays of shape (N, 3) and (M, 3)")
    if cKDTree is None:
        # fallback: brute force distances (O(N*M))
        dists_ab = ((points_a[:, None, :] - points_b[None, :, :]) ** 2).sum(axis=-1)
        dist_a_to_b = np.sqrt(dists_ab.min(axis=1)).mean()
        dist_b_to_a = np.sqrt(dists_ab.min(axis=0)).mean()
    else:
        tree_a = cKDTree(points_a)
        tree_b = cKDTree(points_b)
        # distances from A to B
        dist_a_to_b = np.sqrt(tree_b.query(points_a, k=1)[0] ** 2).mean()
        # distances from B to A
        dist_b_to_a = np.sqrt(tree_a.query(points_b, k=1)[0] ** 2).mean()
    return float(dist_a_to_b + dist_b_to_a) / 2.0


def compute_chamfer_distance(
    predicted: np.ndarray, target: np.ndarray
) -> float:
    """Alias for :func:`chamfer_distance` to maintain backwards compatibility."""
    return chamfer_distance(predicted, target)