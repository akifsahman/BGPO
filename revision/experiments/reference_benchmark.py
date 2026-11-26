"""Full paired benchmark for the manuscript's r=1 comparison.

Each fold is saved independently so interrupted runs can resume without
repeating completed work. Missing or short generator outputs are recorded as
unavailable; they are never silently replaced or compared at another budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import smote_variants as sv
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from benchmark_pilot import build_pool, evaluate
from keel_io import load_fold
from reference_selectors import (
    boundary_indices,
    density_indices,
    diversity_indices,
    random_indices,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "Pool-Select/datasets_keel"
DEFAULT_OUTPUT = ROOT / "revision/experiments/full_run"
METHODS = (
    "Original", "ROS", "SMOTE", "ADASYN", "Borderline_SMOTE1",
    "Borderline_SMOTE2", "MWMOTE", "G_SMOTE", "kmeans_SMOTE",
    "RND", "DIV", "DEN", "BND",
)
CLASSIFIERS = {
    "SVM_RBF": SVC(kernel="rbf", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
}
METRICS = ("F1", "BalancedAccuracy", "GMean", "ROC_AUC", "MCC")
logging.disable(logging.INFO)


def generator_configs(k_safe: int, n_minority: int) -> dict[str, tuple[type, dict]]:
    shared = {"proportion": 1.0, "random_state": 42}
    return {
        "SMOTE": (sv.SMOTE, {**shared, "n_neighbors": k_safe}),
        "ADASYN": (sv.ADASYN, {**shared, "n_neighbors": k_safe}),
        "Borderline_SMOTE1": (
            sv.Borderline_SMOTE1,
            {**shared, "n_neighbors": k_safe, "k_neighbors": k_safe},
        ),
        "Borderline_SMOTE2": (
            sv.Borderline_SMOTE2,
            {**shared, "n_neighbors": k_safe, "k_neighbors": k_safe},
        ),
        "MWMOTE": (
            sv.MWMOTE,
            {**shared, "k1": k_safe, "k2": k_safe, "k3": k_safe},
        ),
        "G_SMOTE": (sv.G_SMOTE, {**shared, "n_neighbors": k_safe}),
        "kmeans_SMOTE": (
            sv.kmeans_SMOTE,
            {**shared, "n_neighbors": k_safe, "n_clusters": min(10, n_minority)},
        ),
    }


def classic_variant(X: np.ndarray, y: np.ndarray, method: str, budget: int) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict]:
    start = time.perf_counter()
    if method == "ROS":
        sampler = RandomOverSampler(
            sampling_strategy={1: int((y == 0).sum())}, random_state=42
        )
        try:
            X_aug, y_aug = sampler.fit_resample(X, y)
        except Exception as exc:
            return None, {"status": "failed", "reason": str(exc), "seconds": time.perf_counter() - start}
    else:
        k_safe = min(5, int((y == 1).sum()) - 1)
        generator_class, parameters = generator_configs(k_safe, int((y == 1).sum()))[method]
        try:
            X_aug, y_aug = generator_class(**parameters).sample(X, y)
        except Exception as exc:
            return None, {
                "status": "failed", "reason": str(exc),
                "parameters": parameters, "seconds": time.perf_counter() - start,
            }
        if (
            len(X_aug) < len(X)
            or not np.array_equal(X_aug[:len(X)], X)
            or not np.array_equal(y_aug[:len(y)], y)
        ):
            return None, {
                "status": "invalid_original_prefix",
                "parameters": parameters,
                "seconds": time.perf_counter() - start,
            }
    added = int((y_aug == 1).sum() - (y == 1).sum())
    diagnostics = {"status": "ok", "added": added, "seconds": time.perf_counter() - start}
    if method != "ROS":
        diagnostics["parameters"] = parameters
    if added != budget:
        diagnostics["status"] = "count_mismatch"
        diagnostics["reason"] = f"Added {added}; required {budget}"
        return None, diagnostics
    return (X_aug, y_aug), diagnostics


def run_fold(dataset_root: Path, name: str, number: int) -> tuple[pd.DataFrame, dict]:
    start_fold = time.perf_counter()
    fold = load_fold(dataset_root, name, number)
    X, y = fold.X_train, fold.y_train
    budget = int((y == 0).sum() - (y == 1).sum())
    if budget <= 0:
        raise ValueError("Fold is not imbalanced")
    variants: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"Original": (X, y)}
    diagnostics: dict[str, dict] = {"Original": {"status": "ok", "seconds": 0.0}}

    pool_start = time.perf_counter()
    pool, pool_generators = build_pool(X, y)
    pool_seconds = time.perf_counter() - pool_start
    if len(pool) < budget:
        raise ValueError(f"Composite pool has {len(pool)} candidates for budget {budget}")

    for method in METHODS[1:9]:
        variants[method], diagnostics[method] = classic_variant(X, y, method, budget)

    selectors = {
        "RND": lambda: random_indices(pool, budget),
        "DIV": lambda: diversity_indices(pool, budget),
        "DEN": lambda: density_indices(pool, budget),
        "BND": lambda: boundary_indices(pool, X[y == 1], X[y == 0], budget),
    }
    for method, selector in selectors.items():
        start = time.perf_counter()
        try:
            indices = selector()
            if len(indices) != budget or len(np.unique(indices)) != budget:
                raise ValueError("Selector violated the budget")
            variants[method] = (
                np.vstack([X, pool[indices]]),
                np.concatenate([y, np.ones(budget, dtype=int)]),
            )
            diagnostics[method] = {"status": "ok", "seconds": time.perf_counter() - start}
        except Exception as exc:
            variants[method] = None
            diagnostics[method] = {
                "status": "failed", "reason": str(exc),
                "seconds": time.perf_counter() - start,
            }

    rows = []
    for method in METHODS:
        variant = variants[method]
        for classifier, template in CLASSIFIERS.items():
            row = {
                "dataset": name,
                "fold": number,
                "method": method,
                "classifier": classifier,
                "status": diagnostics[method]["status"],
                "reason": diagnostics[method].get("reason", ""),
                "train_n": len(y),
                "test_n": len(fold.y_test),
                "minority_before": int((y == 1).sum()),
                "majority_before": int((y == 0).sum()),
                "budget": budget,
                "pool_size": len(pool),
                "minority_after": int((variant[1] == 1).sum()) if variant else np.nan,
                "generation_seconds": (
                    pool_seconds if method in selectors else diagnostics[method]["seconds"]
                ),
                "selection_seconds": diagnostics[method]["seconds"] if method in selectors else 0.0,
                **{metric: np.nan for metric in METRICS},
                "fit_seconds": np.nan,
            }
            if variant is not None:
                try:
                    row.update(evaluate(
                        template, variant[0], variant[1], fold.X_test, fold.y_test
                    ))
                except Exception as exc:
                    row["status"] = "classifier_error"
                    row["reason"] = str(exc)
            rows.append(row)
    report = {
        "dataset": name,
        "fold": number,
        "budget": budget,
        "pool_size": len(pool),
        "nominal_columns": fold.nominal_columns,
        "pool_generators": pool_generators,
        "methods": diagnostics,
        "total_seconds": time.perf_counter() - start_fold,
    }
    return pd.DataFrame(rows), report


def atomic_write_text(path: Path, content: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content)
    os.replace(temp, path)


def aggregate(output: Path) -> pd.DataFrame:
    files = sorted((output / "folds").glob("*.csv"))
    if not files:
        raise ValueError("No fold outputs")
    frames = [pd.read_csv(file) for file in files]
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["dataset", "fold", "method", "classifier"])
    result.to_csv(output / "fold_results.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", nargs="*", help="Optional dataset names for a pilot")
    parser.add_argument("--force", action="store_true", help="Recompute completed fold files")
    args = parser.parse_args()
    output = args.output
    (output / "folds").mkdir(parents=True, exist_ok=True)
    (output / "diagnostics").mkdir(parents=True, exist_ok=True)
    protocol = {
        "augmentation_rate": 1.0,
        "seed": 42,
        "preprocessing": "train-fitted ordinal encoding, median imputation, StandardScaler",
        "classifier_templates": {name: str(template) for name, template in CLASSIFIERS.items()},
        "methods": METHODS,
        "metrics": METRICS,
        "versions": {
            package: metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "smote-variants")
        },
    }
    atomic_write_text(output / "protocol.json", json.dumps(protocol, indent=2) + "\n")
    folders = sorted(folder for folder in args.dataset_root.iterdir() if folder.is_dir())
    if args.only:
        folders = [folder for folder in folders if folder.name in args.only]
    total = len(folders) * 5
    done = 0
    for folder in folders:
        for number in range(1, 6):
            stem = f"{folder.name}__fold{number}"
            fold_csv = output / "folds" / f"{stem}.csv"
            diag_json = output / "diagnostics" / f"{stem}.json"
            if fold_csv.exists() and diag_json.exists() and not args.force:
                done += 1
                continue
            frame, diagnostics = run_fold(args.dataset_root, folder.name, number)
            temp_csv = fold_csv.with_suffix(".csv.tmp")
            frame.to_csv(temp_csv, index=False)
            os.replace(temp_csv, fold_csv)
            atomic_write_text(diag_json, json.dumps(diagnostics, indent=2) + "\n")
            done += 1
            failures = int((frame["status"] != "ok").sum())
            print(
                f"{done}/{total} {folder.name} fold {number}: "
                f"{len(frame)} rows, {failures} unavailable, "
                f"{diagnostics['total_seconds']:.1f}s",
                flush=True,
            )
    result = aggregate(output)
    print(f"Finished {done}/{total} folds, {len(result)} rows -> {output / 'fold_results.csv'}")


if __name__ == "__main__":
    main()
