"""Equal-budget generator-pool ablation and train-only overlap audit.

Nine prespecified pools are evaluated with the *unchanged* BND selector:
four single generators, four leave-one-out mixtures, and the full mixture.
Every successful variant adds exactly the same number of minority candidates.
The KEEL test folds are used only for classifier evaluation, never for pool
construction, overlap measurement, or method selection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import smote_variants as sv
from sklearn.neighbors import NearestNeighbors

from benchmark_pilot import evaluate
from keel_io import load_fold
from reference_benchmark import CLASSIFIERS, METRICS, ROOT, generator_configs
from reference_selectors import boundary_indices


DATA = ROOT / "Pool-Select/datasets_keel"
BASELINE = ROOT / "revision/experiments/full_run/fold_results.csv"
OUTPUT = ROOT / "revision/experiments/pool_overlap_run"
GENERATORS = ("SMOTE", "ADASYN", "MWMOTE", "G_SMOTE")
POOLS = {
    **{f"ONLY_{name}": (name,) for name in GENERATORS},
    **{f"DROP_{name}": tuple(other for other in GENERATORS if other != name)
       for name in GENERATORS},
    "FULL": GENERATORS,
}


def generate_parts(X: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    k_safe = min(5, int((y == 1).sum()) - 1)
    configs = generator_configs(k_safe, int((y == 1).sum()))
    parts: dict[str, np.ndarray] = {}
    diagnostics = {}
    for name in GENERATORS:
        generator_class, parameters = configs[name]
        started = time.perf_counter()
        try:
            X_aug, y_aug = generator_class(**parameters).sample(X, y)
            if (len(X_aug) < len(X) or not np.array_equal(X_aug[:len(X)], X)
                    or not np.array_equal(y_aug[:len(y)], y)):
                raise ValueError("Generator did not preserve original training prefix")
            candidate = X_aug[len(X):][y_aug[len(y):] == 1]
            if not np.isfinite(candidate).all():
                raise ValueError("Generator returned nonfinite candidate")
            parts[name] = candidate
            diagnostics[name] = {
                "status": "ok" if len(candidate) else "empty", "candidates": len(candidate),
                "parameters": parameters, "seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            parts[name] = np.empty((0, X.shape[1]))
            diagnostics[name] = {
                "status": "failed", "candidates": 0, "error": str(exc),
                "parameters": parameters, "seconds": time.perf_counter() - started,
            }
    return parts, diagnostics


def overlap_measurements(X: np.ndarray, y: np.ndarray, pool: np.ndarray) -> dict:
    """All neighborhoods are fitted on the original training fold only."""
    k = min(10, len(X) - 1)
    if k < 1:
        raise ValueError("Too few original training rows")
    model = NearestNeighbors(n_neighbors=k + 1).fit(X)
    # Each original minority point has itself as its nearest neighbor; remove
    # its own index explicitly, including when another point is coincident.
    original_idx = np.flatnonzero(y == 1)
    neighbor_idx = model.kneighbors(X[original_idx], return_distance=False)
    minority_majority_share = []
    for own, row in zip(original_idx, neighbor_idx):
        others = row[row != own][:k]
        if len(others) != k:
            raise ValueError("Could not exclude original point from neighborhood")
        minority_majority_share.append(float((y[others] == 0).mean()))
    # A second, class-conditional measure avoids treating the training prior
    # itself as overlap: compare distances to the two classes at minority
    # observations. The minority point itself is excluded by index.
    minority = X[y == 1]
    majority = X[y == 0]
    k_conditional = min(5, len(minority) - 1, len(majority))
    if k_conditional < 1:
        raise ValueError("Too few observations for class-conditional distance")
    minority_model = NearestNeighbors(n_neighbors=k_conditional + 1).fit(minority)
    same_distances, same_indices = minority_model.kneighbors(minority)
    d_min = np.empty(len(minority))
    for own, distances, indices in zip(range(len(minority)), same_distances, same_indices):
        others = distances[indices != own][:k_conditional]
        if len(others) != k_conditional:
            raise ValueError("Could not exclude minority point from own class")
        d_min[own] = others.mean()
    d_maj = NearestNeighbors(n_neighbors=k_conditional).fit(majority).kneighbors(
        minority
    )[0].mean(axis=1)
    candidate_neighbors = NearestNeighbors(n_neighbors=min(10, len(X))).fit(X).kneighbors(
        pool, return_distance=False
    )
    local_majority = (y[candidate_neighbors] == 0).mean(axis=1)
    return {
        "minority_overlap_mean": float(np.mean(minority_majority_share)),
        "minority_overlap_median": float(np.median(minority_majority_share)),
        "minority_majority_closer_share": float((d_maj < d_min).mean()),
        "minority_distance_ratio_median": float(np.median(d_maj / (d_min + 1e-9))),
        "candidate_local_majority": local_majority,
    }


def run_fold(name: str, number: int, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    fold = load_fold(DATA, name, number)
    X, y = fold.X_train, fold.y_train
    budget = int((y == 0).sum() - (y == 1).sum())
    parts, generators = generate_parts(X, y)
    full_pool = np.vstack([parts[g] for g in GENERATORS])
    if len(full_pool) < budget:
        raise ValueError("Full composite pool misses the budget")
    overlap = overlap_measurements(X, y, full_pool)
    full_local_majority = overlap.pop("candidate_local_majority")
    labels = np.concatenate([np.repeat(g, len(parts[g])) for g in GENERATORS])
    rows = []
    pool_reports = {}
    for method, members in POOLS.items():
        pool = np.vstack([parts[g] for g in members])
        if len(pool) < budget:
            pool_reports[method] = {"status": "pool_short", "pool_size": len(pool)}
            for classifier in CLASSIFIERS:
                rows.append({
                    "dataset": name, "fold": number, "method": method,
                    "classifier": classifier, "status": "pool_short", "budget": budget,
                    "pool_size": len(pool), **{metric: np.nan for metric in METRICS},
                    "fit_seconds": np.nan,
                })
            continue
        selection_started = time.perf_counter()
        selected = boundary_indices(pool, X[y == 1], X[y == 0], budget)
        selection_seconds = time.perf_counter() - selection_started
        if len(selected) != budget or len(np.unique(selected)) != budget:
            raise ValueError("Selector violated the fixed budget")
        # The full pool preserves generator concatenation order; this is the
        # same BND path as the primary benchmark and must reproduce it exactly.
        if method == "FULL":
            shares = full_local_majority[selected]
            selected_labels = labels[selected]
            pool_reports[method] = {
                "status": "ok", "pool_size": len(pool),
                "selected_by_generator": {g: int((selected_labels == g).sum()) for g in GENERATORS},
                "selected_mean_local_majority_share": float(shares.mean()),
                "selected_majority_dominated_share": float((shares > 0.5).mean()),
                "selected_ambiguous_share": float(((shares >= 0.3) & (shares <= 0.7)).mean()),
                "selection_seconds": selection_seconds,
            }
        else:
            pool_reports[method] = {
                "status": "ok", "pool_size": len(pool),
                "selection_seconds": selection_seconds,
            }
        X_aug = np.vstack([X, pool[selected]])
        y_aug = np.concatenate([y, np.ones(budget, dtype=int)])
        for classifier, template in CLASSIFIERS.items():
            scores = evaluate(template, X_aug, y_aug, fold.X_test, fold.y_test)
            if method == "FULL":
                expected = baseline[
                    (baseline.dataset == name) & (baseline.fold == number)
                    & (baseline.method == "BND") & (baseline.classifier == classifier)
                ]
                if len(expected) != 1 or expected.iloc[0].status != "ok":
                    raise ValueError("Missing BND reference row")
                for metric in METRICS:
                    if not np.isclose(scores[metric], expected.iloc[0][metric], rtol=1e-10, atol=1e-12):
                        raise AssertionError(
                            f"Reference drift {name} fold {number} {classifier} {metric}"
                        )
            rows.append({
                "dataset": name, "fold": number, "method": method,
                "classifier": classifier, "status": "ok", "budget": budget,
                "pool_size": len(pool), **scores,
            })
    report = {
        "dataset": name, "fold": number, "budget": budget,
        "generator_diagnostics": generators,
        "pool_reports": pool_reports,
        "minority_overlap_mean": overlap["minority_overlap_mean"],
        "minority_overlap_median": overlap["minority_overlap_median"],
        "minority_majority_closer_share": overlap["minority_majority_closer_share"],
        "minority_distance_ratio_median": overlap["minority_distance_ratio_median"],
        "full_pool_mean_local_majority_share": float(full_local_majority.mean()),
        "full_pool_majority_dominated_share": float((full_local_majority > 0.5).mean()),
        "full_pool_ambiguous_share": float(((full_local_majority >= 0.3) & (full_local_majority <= 0.7)).mean()),
        "reference_full_metric_match": True,
        "seconds": time.perf_counter() - started,
    }
    return pd.DataFrame(rows), report


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--only", nargs="*", help="Dataset names for a pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output
    for folder in (output / "folds", output / "diagnostics"):
        folder.mkdir(parents=True, exist_ok=True)
    protocol = {
        "generators": GENERATORS, "pools": POOLS,
        "selector": "unchanged deterministic BND distance score and top-budget indices",
        "budget": "train-fold majority count minus minority count",
        "overlap": "ten-neighbor local majority share and five-neighbor class-conditional distance ratio, original minority points exclude self; training data only",
        "seed_preprocessing_classifiers_metrics": "identical to full_run/protocol.json",
        "reference_full_metrics_verified_per_fold": True,
    }
    atomic_text(output / "protocol.json", json.dumps(protocol, indent=2) + "\n")
    baseline = pd.read_csv(BASELINE)
    datasets = sorted(folder.name for folder in DATA.iterdir() if folder.is_dir())
    if args.only:
        datasets = [name for name in datasets if name in args.only]
    total = len(datasets) * 5
    done = 0
    for name in datasets:
        for number in range(1, 6):
            stem = f"{name}__fold{number}"
            csv = output / "folds" / f"{stem}.csv"
            diag = output / "diagnostics" / f"{stem}.json"
            if not args.force and csv.exists() and diag.exists():
                done += 1
                continue
            frame, report = run_fold(name, number, baseline)
            temporary = csv.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, csv)
            atomic_text(diag, json.dumps(report, indent=2) + "\n")
            done += 1
            print(f"{done}/{total} {name} fold {number}: "
                  f"majority-closer {report['minority_majority_closer_share']:.3f}, "
                  f"{report['seconds']:.1f}s", flush=True)
    expected_files = [output / "folds" / f"{name}__fold{number}.csv"
                      for name in datasets for number in range(1, 6)]
    if not all(path.exists() for path in expected_files):
        raise ValueError("At least one expected fold file is missing")
    frames = [pd.read_csv(path) for path in expected_files]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output / "fold_results.csv", index=False)
    print(f"Finished {total} folds, {len(result)} classifier rows")


if __name__ == "__main__":
    main()
