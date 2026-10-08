"""K-means clustering.

This module holds TWO implementations:

1. ``KMeansReference`` — a thin wrapper around scikit-learn's KMeans. It works
   out of the box so the pipeline runs end-to-end before you write any code,
   and it gives you a **correctness oracle** to check your own results against.

2. ``KMeansScratch`` — YOUR from-scratch implementation. The methods below are
   stubs. Filling them in is the core of Project 2.

Per the handout: the library version is for *checking* only. Your submitted run
must use ``kmeans.implementation: scratch`` in config.yaml.
"""
from __future__ import annotations

import numpy as np

from .distances import euclidean_sqdist, mahalanobis_sqdist


# --------------------------------------------------------------------------
# Provided baseline — do not submit results from this one.
# --------------------------------------------------------------------------
class KMeansReference:
    """scikit-learn KMeans, wrapped in the same interface as KMeansScratch."""

    def __init__(self, k=3, init="kmeans++", n_init=10, max_iter=300, tol=1e-4,
                 seed=42, **_ignored):
        self.k, self.n_init, self.max_iter, self.tol, self.seed = k, n_init, max_iter, tol, seed
        self.init = "k-means++" if init in ("kmeans++", "k-means++") else "random"
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X: np.ndarray) -> "KMeansReference":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=self.k, init=self.init, n_init=self.n_init,
                    max_iter=self.max_iter, tol=self.tol, random_state=self.seed)
        km.fit(X)
        self.centroids_ = km.cluster_centers_
        self.labels_ = km.labels_.astype("int64")
        self.inertia_ = float(km.inertia_)
        self.n_iter_ = int(km.n_iter_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return euclidean_sqdist(X, self.centroids_).argmin(axis=1).astype("int64")


# --------------------------------------------------------------------------
# YOUR implementation
# --------------------------------------------------------------------------
class KMeansScratch:
    """K-means (Lloyd's algorithm), implemented from scratch.

    Attributes set by ``fit``:
        centroids_ : (k, d) final centroids
        labels_    : (n,)   cluster index of each point
        inertia_   : float  within-cluster sum of squared distances (the objective J)
        n_iter_    : int    iterations actually run

    Supports two distance metrics, selected at construction:
        "euclidean"   — plain squared Euclidean
        "mahalanobis" — simplified diagonal Mahalanobis (Task 4)
    """

    def __init__(self, k=3, init="kmeans++", n_init=10, max_iter=300, tol=1e-4,
                 seed=42, distance="euclidean", mahalanobis_diag=None):
        self.k = k
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.distance = distance
        self.mahalanobis_diag = mahalanobis_diag
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # -- distance dispatch (provided) --------------------------------------
    def _sqdist(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """(n, k) squared distances under the configured metric."""
        if self.distance == "euclidean":
            return euclidean_sqdist(X, C)
        if self.distance == "mahalanobis":
            diag = self.mahalanobis_diag
            diag = np.ones(X.shape[1]) if diag is None else np.asarray(diag, dtype="float64")
            if diag.shape != (X.shape[1],):
                raise ValueError(
                    f"mahalanobis_diag must have length {X.shape[1]}, got {diag.shape}"
                )
            if np.any(diag <= 0):
                raise ValueError("mahalanobis_diag entries must be strictly positive")
            return mahalanobis_sqdist(X, C, diag)
        raise ValueError(f"unknown distance {self.distance!r}")

    # -- the three steps you implement -------------------------------------
    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]

        if self.init == "random":
            indices = rng.choice(n, size=self.k, replace=False)
            return X[indices].copy()

        elif self.init == "kmeans++":
            first_idx = rng.integers(n)
            centroids = [X[first_idx].copy()]

            selected_indices = [first_idx]

            while len(centroids) < self.k:
                current_centroids = np.array(centroids)


                distances = self._sqdist(X, current_centroids)

                nearest_dist = np.min(distances, axis=1)

                total_dist = np.sum(nearest_dist)

                if total_dist == 0:
                    remaining = [
                        i for i in range(n)
                        if i not in selected_indices
                    ]

                    next_idx = rng.choice(remaining)

                else:
                    probabilities = nearest_dist / total_dist
                    next_idx = rng.choice(n, p=probabilities)

                selected_indices.append(next_idx)
                centroids.append(X[next_idx].copy())

            return np.array(centroids)

        else:
            raise ValueError(f"Unknown initialization method: {self.init}")




    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:

        distances = self._sqdist(X, centroids)
        labels = np.argmin(distances, axis=1).astype("int64")
        return labels

    def _update(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:

        new_centroids = centroids.copy()

        for i in range(self.k):
            cluster_p = X[labels == i]

            if cluster_p.shape[0] > 0:
                new_centroids[i] = np.mean(cluster_p, axis=0)

        return new_centroids

    # -- the fit loop you implement ----------------------------------------
    def fit(self, X: np.ndarray) -> "KMeansScratch":
        rng = np.random.default_rng(self.seed)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = None

        for _ in range(self.n_init):

            centroids = self._init_centroids(X, rng)

            old_labels = None

            for iteration in range(1, self.max_iter + 1):

                labels = self._assign(X, centroids)

                new_centroids = self._update(X, labels, centroids)

                shifts = np.linalg.norm(new_centroids - centroids, axis=1)
                max_shift = np.max(shifts)

                labels_unchanged = (
                    old_labels is not None
                    and np.array_equal(labels, old_labels)
                )

                centroids = new_centroids

                if max_shift < self.tol or labels_unchanged:
                    break

                old_labels = labels.copy()

            labels = self._assign(X, centroids)

            distances = self._sqdist(X, centroids)

            inertia = np.sum(
                distances[np.arange(X.shape[0]), labels]
            )

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()
                best_n_iter = iteration

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the fitted centroids."""
        if self.centroids_ is None:
            raise RuntimeError("call fit() before predict()")
        return self._assign(X, self.centroids_)


def build(cfg: dict):
    """Construct the clusterer selected by config.yaml."""
    kc = cfg["kmeans"]
    impl = kc.get("implementation", "sklearn")
    common = dict(k=kc["k"], init=kc.get("init", "kmeans++"),
                  n_init=kc.get("n_init", 10), max_iter=kc.get("max_iter", 300),
                  tol=float(kc.get("tol", 1e-4)), seed=cfg["seed"])
    if impl == "sklearn":
        if kc.get("distance", "euclidean") != "euclidean":
            raise ValueError("The sklearn baseline supports Euclidean only; use scratch for Mahalanobis.")
        return KMeansReference(**common)
    if impl == "scratch":
        return KMeansScratch(distance=kc.get("distance", "euclidean"),
                             mahalanobis_diag=kc.get("mahalanobis_diag"), **common)
    raise ValueError(f"unknown kmeans.implementation {impl!r}")
