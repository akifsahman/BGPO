"""Selection rules matching the working manuscript.

All functions return candidate indices. Candidate rows are kept by index even
when two generators produce identical coordinates. Pool generation, preprocessing,
and classifiers are separate parts of the benchmark.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def _validate(pool: np.ndarray, budget: int) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.float64)
    if pool.ndim != 2 or len(pool) == 0 or not np.isfinite(pool).all():
        raise ValueError("The pool must be a nonempty finite 2D array")
    if not 0 <= budget <= len(pool):
        raise ValueError("The selection budget must not exceed the pool size")
    return pool


def _top_indices(scores: np.ndarray, budget: int) -> np.ndarray:
    """Descending score order, with ascending candidate index breaking ties."""
    if not np.isfinite(scores).all():
        raise ValueError("Selection scores must be finite")
    return np.lexsort((np.arange(len(scores)), -scores))[:budget]


def random_indices(pool: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    pool = _validate(pool, budget)
    return np.random.default_rng(seed).choice(len(pool), size=budget, replace=False)


def density_scores(pool: np.ndarray, k: int = 5, epsilon: float = 1e-9) -> np.ndarray:
    pool = _validate(pool, 0)
    if len(pool) < 2:
        raise ValueError("Density selection needs at least two pool candidates")
    n_neighbors = min(len(pool), k + 1)
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors).fit(pool).kneighbors(pool)
    effective_k = min(k, len(pool) - 1)
    means = np.empty(len(pool))
    for i, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        other = row_distances[row_indices != i][:effective_k]
        if len(other) != effective_k:
            raise RuntimeError("Could not exclude the candidate's own index")
        means[i] = other.mean()
    return 1.0 / (means + epsilon)


def density_indices(pool: np.ndarray, budget: int, k: int = 5) -> np.ndarray:
    pool = _validate(pool, budget)
    if budget == 0:
        return np.empty(0, dtype=int)
    return _top_indices(density_scores(pool, k=k), budget)


def boundary_scores(
    pool: np.ndarray,
    minority: np.ndarray,
    majority: np.ndarray,
    k: int = 5,
    epsilon: float = 1e-9,
) -> np.ndarray:
    pool = _validate(pool, 0)
    minority = np.asarray(minority, dtype=np.float64)
    majority = np.asarray(majority, dtype=np.float64)
    if minority.ndim != 2 or majority.ndim != 2:
        raise ValueError("Both original class arrays must be 2D")
    if not len(minority) or not len(majority):
        raise ValueError("Both original classes must have observations")
    if pool.shape[1] != minority.shape[1] or pool.shape[1] != majority.shape[1]:
        raise ValueError("Pool and training data have different feature counts")
    d_min = NearestNeighbors(n_neighbors=min(k, len(minority))).fit(minority).kneighbors(pool)[0].mean(axis=1)
    d_maj = NearestNeighbors(n_neighbors=min(k, len(majority))).fit(majority).kneighbors(pool)[0].mean(axis=1)
    return 1.0 / ((np.abs(d_min - d_maj) + epsilon) * (d_min + d_maj + epsilon))


def boundary_indices(
    pool: np.ndarray,
    minority: np.ndarray,
    majority: np.ndarray,
    budget: int,
    k: int = 5,
) -> np.ndarray:
    pool = _validate(pool, budget)
    if budget == 0:
        return np.empty(0, dtype=int)
    return _top_indices(boundary_scores(pool, minority, majority, k=k), budget)


def _proportional_quotas(sizes: np.ndarray, budget: int) -> np.ndarray:
    """Largest-remainder allocation, with cluster index breaking ties."""
    exact = budget * sizes / sizes.sum()
    quotas = np.floor(exact).astype(int)
    remaining = budget - quotas.sum()
    order = np.lexsort((np.arange(len(sizes)), -(exact - quotas)))
    for cluster in order:
        if remaining == 0:
            break
        if quotas[cluster] < sizes[cluster]:
            quotas[cluster] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Cluster quotas could not meet the budget")
    return quotas


def diversity_indices(pool: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    pool = _validate(pool, budget)
    if budget == 0:
        return np.empty(0, dtype=int)
    if budget == len(pool):
        return np.arange(len(pool))
    k_clusters = min(max(2, budget // 5), max(1, len(pool) // 2), 20)
    model = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(pool)
    clusters = np.unique(labels)
    sizes = np.array([(labels == c).sum() for c in clusters])
    quotas = _proportional_quotas(sizes, budget)
    selected: list[int] = []
    for c, quota in zip(clusters, quotas):
        if quota == 0:
            continue
        members = np.flatnonzero(labels == c)
        centroid = model.cluster_centers_[c]
        first = members[np.argmin(np.linalg.norm(pool[members] - centroid, axis=1))]
        chosen = [int(first)]
        nearest = np.linalg.norm(pool[members] - pool[first], axis=1)
        while len(chosen) < quota:
            available = ~np.isin(members, chosen)
            next_index = members[available][np.argmax(nearest[available])]
            chosen.append(int(next_index))
            nearest = np.minimum(nearest, np.linalg.norm(pool[members] - pool[next_index], axis=1))
        selected.extend(chosen)
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError("Diversity selection violated the budget")
    return np.asarray(selected, dtype=int)


def boundary_quota_indices(
    pool: np.ndarray,
    minority: np.ndarray,
    majority: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Select top boundary scores within DIV's proportional spatial quotas.

    This exploratory rule changes only the allocation of the fixed budget:
    the composite pool, original BND score, cluster count, and DIV quota rule
    are unchanged. It has no fitted weight or test-fold-tuned threshold.
    """
    pool = _validate(pool, budget)
    if budget == 0:
        return np.empty(0, dtype=int)
    if budget == len(pool):
        return np.arange(len(pool))
    scores = boundary_scores(pool, minority, majority)
    k_clusters = min(max(2, budget // 5), max(1, len(pool) // 2), 20)
    labels = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10).fit_predict(pool)
    clusters = np.unique(labels)
    sizes = np.array([(labels == c).sum() for c in clusters])
    quotas = _proportional_quotas(sizes, budget)
    selected: list[int] = []
    for cluster, quota in zip(clusters, quotas):
        if quota == 0:
            continue
        members = np.flatnonzero(labels == cluster)
        order = np.lexsort((members, -scores[members]))
        selected.extend(members[order[:quota]].tolist())
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError("Boundary quota selection violated the budget")
    return np.asarray(selected, dtype=int)
