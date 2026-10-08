"""Distance functions used by the clustering algorithms.

`euclidean_sqdist` is provided as a worked example of the vectorized NumPy style
we want. `mahalanobis_sqdist` is YOURS to implement (Project 2, Task 4).
"""
from __future__ import annotations

import numpy as np


def euclidean_sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every point to every centroid.

    Parameters
    ----------
    X : (n, d) data
    C : (k, d) centroids

    Returns
    -------
    (n, k) array where entry [i, j] is ||x_i - c_j||^2.

    We square-and-sum the broadcast difference. Squared distance is enough for
    K-means: sqrt is monotonic, so argmin is unchanged, and we skip the cost.
    """
    diff = X[:, None, :] - C[None, :, :]          # (n, k, d)
    return np.einsum("nkd,nkd->nk", diff, diff)   # (n, k)


def mahalanobis_sqdist(X: np.ndarray, C: np.ndarray, diag: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - C[None, :, :]

    squared_diff = diff ** 2

    weighted_squared_diff = squared_diff * diag

    return np.sum(weighted_squared_diff, axis=2)
