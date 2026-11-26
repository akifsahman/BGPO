"""Run the frozen, independent three-dataset validation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from benchmark_pilot import build_pool, evaluate
from reference_benchmark import CLASSIFIERS, METRICS, ROOT, classic_variant
from reference_selectors import boundary_indices, boundary_quota_indices, diversity_indices


DATA = ROOT / "revision/experiments/external_data"
OUTPUT = ROOT / "revision/experiments/external_run"
NAMES = ("mammography", "MagicTelescope", "satimage")
METHODS = ("Original", "SMOTE", "BND", "DIV", "BND_QUOTA")


def run_fold(name: str, fold: int, train: np.ndarray, test: np.ndarray,
             X_raw: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_raw[train]))
    X_test = scaler.transform(imputer.transform(X_raw[test]))
    y_train, y_test = y[train], y[test]
    budget = int((y_train == 0).sum() - (y_train == 1).sum())
    if budget <= 0:
        raise ValueError("Training fold has no positive augmentation budget")
    pool, generators = build_pool(X_train, y_train)
    if len(pool) < budget:
        raise ValueError("Pool cannot satisfy fixed augmentation budget")
    variants = {"Original": (X_train, y_train)}
    classic, smote_diag = classic_variant(X_train, y_train, "SMOTE", budget)
    variants["SMOTE"] = classic
    selection = {
        "BND": boundary_indices(pool, X_train[y_train == 1], X_train[y_train == 0], budget),
        "DIV": diversity_indices(pool, budget),
        "BND_QUOTA": boundary_quota_indices(pool, X_train[y_train == 1], X_train[y_train == 0], budget),
    }
    for method, indices in selection.items():
        if len(indices) != budget or len(np.unique(indices)) != budget:
            raise ValueError(f"{method} violated the budget")
        variants[method] = (
            np.vstack([X_train, pool[indices]]),
            np.concatenate([y_train, np.ones(budget, dtype=int)]),
        )
    rows = []
    for method in METHODS:
        variant = variants[method]
        for classifier, template in CLASSIFIERS.items():
            row = {
                "dataset": name, "fold": fold, "method": method,
                "classifier": classifier,
                "status": "ok" if variant is not None else smote_diag["status"],
                "reason": "" if variant is not None else smote_diag.get("reason", ""),
                "train_n": len(train), "test_n": len(test),
                "minority_train": int((y_train == 1).sum()),
                "budget": budget, "pool_size": len(pool),
                **{metric: np.nan for metric in METRICS},
                "fit_seconds": np.nan,
            }
            if variant is not None:
                try:
                    row.update(evaluate(template, *variant, X_test, y_test))
                except Exception as exc:
                    row["status"] = "classifier_error"
                    row["reason"] = str(exc)
            rows.append(row)
    report = {
        "dataset": name, "fold": fold,
        "train_indices_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "train_n": len(train), "test_n": len(test),
        "minority_train": int((y_train == 1).sum()),
        "minority_test": int((y_test == 1).sum()),
        "budget": budget, "pool_size": len(pool),
        "pool_generators": generators, "smote": smote_diag,
        "selected_intersection": int(len(set(selection["BND"]) & set(selection["BND_QUOTA"]))),
        "seconds": time.perf_counter() - started,
    }
    return pd.DataFrame(rows), report


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=NAMES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for directory in (OUTPUT / "folds", OUTPUT / "diagnostics"):
        directory.mkdir(parents=True, exist_ok=True)
    protocol = {
        "source": "external_validation_protocol.md (frozen before outcomes)",
        "dataset_ids": {"mammography": 310, "MagicTelescope": 1120, "satimage": 182},
        "folds": "StratifiedKFold(5, shuffle=True, random_state=20261125)",
        "preprocessing": "training-only median imputation, StandardScaler",
        "methods": METHODS,
        "classifiers": {key: str(value) for key, value in CLASSIFIERS.items()},
        "metrics": METRICS,
    }
    atomic_json(OUTPUT / "protocol.json", protocol)
    names = args.only or NAMES
    for name in names:
        path = DATA / f"{name}.csv"
        metadata = json.loads((DATA / f"{name}.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256_csv"]:
            raise ValueError(f"Raw snapshot checksum mismatch: {name}")
        table = pd.read_csv(path)
        y = table.pop("__target__").to_numpy(dtype=int)
        X_raw = table.to_numpy(dtype=float)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261125)
        for number, (train, test) in enumerate(folds.split(X_raw, y), start=1):
            stem = f"{name}__fold{number}"
            csv = OUTPUT / "folds" / f"{stem}.csv"
            diag = OUTPUT / "diagnostics" / f"{stem}.json"
            if csv.exists() and diag.exists() and not args.force:
                continue
            frame, report = run_fold(name, number, train, test, X_raw, y)
            temp = csv.with_suffix(".csv.tmp")
            frame.to_csv(temp, index=False)
            os.replace(temp, csv)
            atomic_json(diag, report)
            print(f"{name} {number}/5: {report['seconds']:.1f}s, budget {report['budget']}", flush=True)
    expected = [OUTPUT / "folds" / f"{name}__fold{number}.csv"
                for name in NAMES for number in range(1, 6)]
    if all(path.exists() for path in expected):
        combined = pd.concat([pd.read_csv(path) for path in expected], ignore_index=True)
        combined.to_csv(OUTPUT / "fold_results.csv", index=False)
        print(f"Complete: {len(combined)} rows", flush=True)


if __name__ == "__main__":
    main()
