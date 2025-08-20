"""Evaluation metrics for BioViT3R.

This module defines functions for computing quantitative evaluation
metrics used in training and assessing the BioViT3R system. The metrics
include biomass estimation errors and classification accuracy.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    from sklearn.metrics import r2_score, accuracy_score  # type: ignore
except ImportError:
    r2_score = None  # type: ignore
    accuracy_score = None  # type: ignore


def compute_biomass_error(preds: Iterable[float], targets: Iterable[float]) -> Tuple[float, float, float]:
    """Compute MAE, MAPE and R^2 for biomass predictions.

    Args:
        preds: Iterable of predicted biomass values.
        targets: Iterable of ground truth biomass values.

    Returns:
        Tuple ``(mae, mape, r2)`` where:
            - ``mae`` is the mean absolute error.
            - ``mape`` is the mean absolute percentage error (in %).
            - ``r2`` is the coefficient of determination.
    """
    preds_arr = np.array(list(preds), dtype=float)
    targets_arr = np.array(list(targets), dtype=float)
    if preds_arr.shape != targets_arr.shape:
        raise ValueError("preds and targets must have the same shape")
    mae = np.mean(np.abs(preds_arr - targets_arr))
    # avoid division by zero in MAPE
    eps = 1e-8
    mape = np.mean(np.abs((preds_arr - targets_arr) / (targets_arr + eps))) * 100.0
    # R^2 using sklearn if available
    if r2_score is not None:
        r2 = float(r2_score(targets_arr, preds_arr))
    else:
        # manual R^2: 1 - sum((y - y_hat)^2) / sum((y - mean)^2)
        ss_res = np.sum((targets_arr - preds_arr) ** 2)
        ss_tot = np.sum((targets_arr - targets_arr.mean()) ** 2) + eps
        r2 = float(1.0 - ss_res / ss_tot)
    return mae, mape, r2