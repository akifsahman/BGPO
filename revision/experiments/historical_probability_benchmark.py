"""Paired re-evaluation of the historical stochastic boundary rule.

This is a post-hoc method comparison. It holds the corrected benchmark's folds,
train-fitted preprocessing, pool, budget, and classifier templates fixed, while
substituting exactly the historical notebook's boundary score and sampling rule.
The historical notebook itself used different preprocessing and other settings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from benchmark_pilot import build_pool, evaluate
from keel_io import load_fold
from reference_benchmark import CLASSIFIERS, METRICS, ROOT
from reference_selectors import boundary_indices


DATA = ROOT / "Pool-Select/datasets_keel"
BASELINE = ROOT / "revision/experiments/full_run/fold_results.csv"
OUTPUT = ROOT / "revision/experiments/historical_probability_run"


def historical_indices(pool: np.ndarray, X: np.ndarray, y: np.ndarray,
                       budget: int, seed: int = 42) -> np.ndarray:
    """Mirror benchmark_pipeline_pool_ablation.py lines 419--422, 450--462, 545--561."""
    k = min(10, len(X) - 1)
    if k < 1 or budget <= 0 or len(pool) == 0:
        raise ValueError("Invalid historical boundary inputs")
    model = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    _, neighbors = model.kneighbors(pool)
    p_min = (y[neighbors] == 1).mean(axis=1)
    p_maj = (y[neighbors] == 0).mean(axis=1)
    scores = np.clip(1.0 - np.abs(p_min - p_maj), 1e-9, None)
    scores = scores - np.nanmin(scores) + 1e-9
    probabilities = scores / scores.sum()
    return np.random.RandomState(seed).choice(
        len(pool), size=budget, replace=len(pool) < budget, p=probabilities
    )


def run_fold(name: str, number: int, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    fold = load_fold(DATA, name, number)
    X, y = fold.X_train, fold.y_train
    budget = int((y == 0).sum() - (y == 1).sum())
    pool, generators = build_pool(X, y)
    if len(pool) < budget:
        raise ValueError("Pool misses budget")
    deterministic = boundary_indices(pool, X[y == 1], X[y == 0], budget)
    historical = historical_indices(pool, X, y, budget)
    y_aug = np.concatenate([y, np.ones(budget, dtype=int)])
    rows = []
    for classifier, template in CLASSIFIERS.items():
        expected = baseline[(baseline.dataset == name) &
                            (baseline.fold == number) &
                            (baseline.method == "BND") &
                            (baseline.classifier == classifier)]
        if len(expected) != 1 or expected.iloc[0].status != "ok":
            raise ValueError("Missing reference BND result")
        reference = evaluate(template, np.vstack([X, pool[deterministic]]), y_aug,
                             fold.X_test, fold.y_test)
        for metric in METRICS:
            if not np.isclose(reference[metric], expected.iloc[0][metric],
                              rtol=1e-10, atol=1e-12):
                raise AssertionError(f"Reference drift: {name} {number} {classifier} {metric}")
        scores = evaluate(template, np.vstack([X, pool[historical]]), y_aug,
                          fold.X_test, fold.y_test)
        rows.append({
            "dataset": name, "fold": number,
            "method": "HIST_PROB_BND", "classifier": classifier,
            "status": "ok", "budget": budget,
            "pool_size": len(pool), **scores,
        })
    report = {
        "dataset": name, "fold": number, "budget": budget,
        "pool_size": len(pool), "generator_diagnostics": generators,
        "same_candidate_indices": int(len(set(deterministic) & set(historical))),
        "historical_unique_indices": int(len(np.unique(historical))),
        "reference_metric_match": True,
        "seconds": time.perf_counter() - started,
    }
    return pd.DataFrame(rows), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for folder in (OUTPUT / "folds", OUTPUT / "diagnostics"):
        folder.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "post-hoc, because historical outcomes were already seen",
        "single_change_from_corrected_full_run": "10-nearest combined-class proportion score and seeded weighted sampling without replacement, as in historical/notebook/benchmark_pipeline_pool_ablation.py",
        "unchanged": "44 KEEL datasets, five predefined folds, train-fitted preprocessing, four-generator pool, exact budget, three classifier templates, five metrics",
        "historical_seed": 42,
        "reference_BND_recomputed_and_checked_per_fold": True,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    baseline = pd.read_csv(BASELINE)
    names = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    if args.only:
        names = [name for name in names if name in args.only]
    for name in names:
        for number in range(1, 6):
            stem = f"{name}__fold{number}"
            csv = OUTPUT / "folds" / f"{stem}.csv"
            diag = OUTPUT / "diagnostics" / f"{stem}.json"
            if csv.exists() and diag.exists() and not args.force:
                continue
            frame, report = run_fold(name, number, baseline)
            temp = csv.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            os.replace(temp, csv)
            temporary = diag.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            os.replace(temporary, diag)
        print(f"{name}: 5 folds", flush=True)
    expected = [OUTPUT / "folds" / f"{name}__fold{number}.csv"
                for name in names for number in range(1, 6)]
    if all(path.exists() for path in expected):
        result = pd.concat([pd.read_csv(path) for path in expected], ignore_index=True)
        result.to_csv(OUTPUT / "fold_results.csv", index=False)
        print(f"Complete: {len(result)} classifier rows")


if __name__ == "__main__":
    main()
