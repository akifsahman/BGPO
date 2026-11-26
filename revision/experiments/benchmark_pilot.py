"""Small paired benchmark to validate the revised fold and selector pipeline."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import smote_variants as sv
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from keel_io import load_fold
from reference_selectors import (
    boundary_indices,
    density_indices,
    diversity_indices,
    random_indices,
)


ROOT = Path(__file__).resolve().parents[2]
logging.getLogger("smote_variants").setLevel(logging.ERROR)
logging.getLogger("NearestNeighborsWithMetricTensor").setLevel(logging.ERROR)


def build_pool(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    k_safe = min(5, int((y == 1).sum()) - 1)
    if k_safe < 1:
        raise ValueError("Too few minority observations for generators")
    configurations = {
        "SMOTE": (sv.SMOTE, {"proportion": 1.0, "n_neighbors": k_safe, "random_state": 42}),
        "ADASYN": (sv.ADASYN, {"proportion": 1.0, "n_neighbors": k_safe, "random_state": 42}),
        "MWMOTE": (sv.MWMOTE, {"proportion": 1.0, "k1": k_safe, "k2": k_safe, "k3": k_safe, "random_state": 42}),
        "G_SMOTE": (sv.G_SMOTE, {"proportion": 1.0, "n_neighbors": k_safe, "random_state": 42}),
    }
    parts = []
    diagnostics = {}
    for name, (generator_class, parameters) in configurations.items():
        start = time.perf_counter()
        try:
            generator = generator_class(**parameters)
            X_res, y_res = generator.sample(X, y)
            if len(X_res) < len(X) or not np.array_equal(X_res[:len(X)], X) or not np.array_equal(y_res[:len(y)], y):
                raise ValueError("Generator output did not retain the original training rows")
        except Exception as exc:
            diagnostics[name] = {
                "status": "failed",
                "parameters": parameters,
                "error": str(exc),
                "seconds": time.perf_counter() - start,
            }
            continue
        added = X_res[len(X):][y_res[len(y):] == 1]
        diagnostics[name] = {
            "status": "ok" if len(added) else "empty",
            "candidates": len(added),
            "parameters": parameters,
            "seconds": time.perf_counter() - start,
        }
        if len(added):
            parts.append(added)
    if not parts:
        raise ValueError("No generator produced candidates")
    return np.vstack(parts), diagnostics


def evaluate(
    template,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    clf = clone(template)
    start = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    prediction = clf.predict(X_test)
    if hasattr(clf, "decision_function"):
        score = clf.decision_function(X_test)
    else:
        score = clf.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "F1": f1_score(y_test, prediction, zero_division=0),
        "BalancedAccuracy": balanced_accuracy_score(y_test, prediction),
        "GMean": float(np.sqrt(sensitivity * specificity)),
        "ROC_AUC": roc_auc_score(y_test, score),
        "MCC": matthews_corrcoef(y_test, prediction),
        "fit_seconds": fit_seconds,
    }


def run_one(dataset: str, fold_number: int) -> dict:
    fold = load_fold(ROOT / "Pool-Select/datasets_keel", dataset, fold_number)
    X, y = fold.X_train, fold.y_train
    budget = int((y == 0).sum() - (y == 1).sum())
    pool, generators = build_pool(X, y)
    if len(pool) < budget:
        raise ValueError("The composite pool does not meet the selection budget")
    selectors = {
        "RND": lambda: random_indices(pool, budget),
        "DIV": lambda: diversity_indices(pool, budget),
        "DEN": lambda: density_indices(pool, budget),
        "BND": lambda: boundary_indices(pool, X[y == 1], X[y == 0], budget),
    }
    selected = {}
    selection_seconds = {}
    for name, selector in selectors.items():
        start = time.perf_counter()
        selected[name] = selector()
        selection_seconds[name] = time.perf_counter() - start
    variants = {"Original": (X, y)}
    ros = RandomOverSampler(sampling_strategy={1: int((y == 0).sum())}, random_state=42)
    variants["ROS"] = ros.fit_resample(X, y)
    for name, indices in selected.items():
        variants[name] = (
            np.vstack([X, pool[indices]]),
            np.concatenate([y, np.ones(budget, dtype=int)]),
        )
    templates = {
        "SVM_RBF": SVC(kernel="rbf", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    }
    results = []
    for method, (X_aug, y_aug) in variants.items():
        for classifier, template in templates.items():
            scores = evaluate(template, X_aug, y_aug, fold.X_test, fold.y_test)
            results.append({
                "method": method,
                "classifier": classifier,
                "training_minority": int((y_aug == 1).sum()),
                **scores,
            })
    return {
        "dataset": dataset,
        "fold": fold_number,
        "train_size": len(y),
        "test_size": len(fold.y_test),
        "budget": budget,
        "pool_size": len(pool),
        "nominal_columns": fold.nominal_columns,
        "generator_diagnostics": generators,
        "selection_seconds": selection_seconds,
        "results": results,
    }


if __name__ == "__main__":
    payload = {
        "protocol": "train-fitted numeric encoding and StandardScaler; four reference selectors; r=1; one fold per dataset",
        "runs": [run_one("pima", 1), run_one("abalone9-18", 1)],
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for run in payload["runs"]:
        print(
            run["dataset"], "fold", run["fold"],
            "budget", run["budget"], "pool", run["pool_size"],
            "methods", len({r["method"] for r in run["results"]}),
            "classifiers", len({r["classifier"] for r in run["results"]}),
        )
    print(output)
