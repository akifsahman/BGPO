"""Nested, train-only selection among SMOTE, historical BND, historical DIV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from benchmark_pilot import evaluate
from fetch_adaptive_external import OUTPUT as DATA, SOURCES
from pool_overlap_benchmark import GENERATORS, generate_parts
from reference_benchmark import CLASSIFIERS, METRICS, ROOT


sys.path.insert(0, str(ROOT / "historical/notebook"))
from benchmark_pipeline_pool_ablation import (  # noqa: E402
    build_neighbor_context,
    proposed_boundary_from_pool,
    proposed_diversity_from_pool,
)


OUT = ROOT / "revision/experiments/adaptive_external_run"
METHODS = ("SMOTE", "HIST_BND", "HIST_DIV")


def preprocess(X_train_raw: np.ndarray,
               X_valid_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
    X_valid = scaler.transform(imputer.transform(X_valid_raw))
    if not np.isfinite(X_train).all() or not np.isfinite(X_valid).all():
        raise ValueError("Nonfinite features after train-only preprocessing")
    return X_train, X_valid


def make_variants(X: np.ndarray, y: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    budget = int((y == 0).sum() - (y == 1).sum())
    if budget <= 0:
        raise ValueError("No positive augmentation budget")
    parts, generators = generate_parts(X, y)
    pool = np.vstack([parts[g] for g in GENERATORS])
    if len(parts["SMOTE"]) != budget or len(pool) < budget:
        raise ValueError("SMOTE or four-generator pool missed exact budget")
    pool_y = np.ones(len(pool), dtype=int)
    variants = {
        "SMOTE": (np.vstack([X, parts["SMOTE"]]),
                  np.concatenate([y, np.ones(budget, dtype=int)])),
        "HIST_BND": proposed_boundary_from_pool(
            X, y, budget, pool, pool_y, build_neighbor_context(X, y)
        ),
        "HIST_DIV": proposed_diversity_from_pool(
            X, y, budget, pool, pool_y
        ),
    }
    for name, (X_aug, y_aug) in variants.items():
        if (len(X_aug) != len(X) + budget or
                int((y_aug == 1).sum()) != int((y == 0).sum())):
            raise ValueError(f"{name} missed augmentation budget")
    return variants, {"budget": budget, "pool_size": len(pool),
                      "generators": generators}


def load_source(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data_id, version = SOURCES[name]
    path = DATA / f"{name}.csv"
    metadata = json.loads((DATA / f"{name}.json").read_text())
    if (hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256_csv"] or
            metadata["openml_data_id"] != data_id or
            metadata["openml_version"] != version):
        raise ValueError(f"Source snapshot mismatch: {name}")
    table = pd.read_csv(path)
    y = table.pop("__target__").to_numpy(dtype=int)
    X = table.to_numpy(dtype=float)
    if not np.isfinite(y).all() or len(np.unique(y)) != 2:
        raise ValueError(f"Invalid target in {name}")
    return X, y, metadata


def inner_decision(X_raw: np.ndarray, y: np.ndarray,
                   name: str, outer_number: int) -> tuple[dict[str, str], pd.DataFrame, dict]:
    started = time.perf_counter()
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261126)
    records = []
    for inner_number, (train, valid) in enumerate(splitter.split(X_raw, y), start=1):
        X_train, X_valid = preprocess(X_raw[train], X_raw[valid])
        variants, info = make_variants(X_train, y[train])
        for method in METHODS:
            X_aug, y_aug = variants[method]
            for classifier, template in CLASSIFIERS.items():
                scores = evaluate(template, X_aug, y_aug, X_valid, y[valid])
                records.append({"dataset": name, "outer_fold": outer_number,
                                "inner_fold": inner_number, "method": method,
                                "classifier": classifier, "budget": info["budget"],
                                "pool_size": info["pool_size"], **scores})
    inner = pd.DataFrame(records)
    if len(inner) != 3 * len(METHODS) * len(CLASSIFIERS):
        raise ValueError("Incomplete inner validation")
    choices = {}
    inner_means = {}
    for classifier in CLASSIFIERS:
        subset = inner[inner.classifier == classifier]
        scores = subset.groupby("method").GMean.agg(["mean", "count"])
        if set(scores.index) != set(METHODS) or (scores["count"] != 3).any():
            raise ValueError("Incomplete method scores for inner choice")
        ordered_scores = [float(scores.loc[method, "mean"]) for method in METHODS]
        choices[classifier] = METHODS[int(np.argmax(ordered_scores))]
        inner_means[classifier] = {
            method: float(scores.loc[method, "mean"]) for method in METHODS
        }
    details = {"dataset": name, "outer_fold": outer_number,
               "decision_seconds": time.perf_counter() - started,
               "selection_rule": "highest mean inner GMean; exact ties SMOTE, HIST_BND, HIST_DIV",
               "choices": choices, "inner_mean_GMean": inner_means}
    return choices, inner, details


def run_fold(name: str, number: int, X_raw: np.ndarray, y: np.ndarray,
             train: np.ndarray, test: np.ndarray) -> None:
    stem = f"{name}__fold{number}"
    result_path = OUT / "folds" / f"{stem}.csv"
    choice_path = OUT / "choices" / f"{stem}.json"
    inner_path = OUT / "inner" / f"{stem}.csv"
    diagnostic_path = OUT / "diagnostics" / f"{stem}.json"
    if result_path.exists() and choice_path.exists() and inner_path.exists() and diagnostic_path.exists():
        return
    if choice_path.exists():
        details = json.loads(choice_path.read_text())
        choices = details["choices"]
        if not inner_path.exists():
            raise ValueError("Saved choice lacks inner-score audit")
    else:
        choices, inner, details = inner_decision(X_raw[train], y[train], name, number)
        temporary = inner_path.with_suffix(".csv.tmp")
        inner.to_csv(temporary, index=False)
        os.replace(temporary, inner_path)
        temporary = choice_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(details, indent=2) + "\n")
        os.replace(temporary, choice_path)
    X_train, X_test = preprocess(X_raw[train], X_raw[test])
    variants, variant_info = make_variants(X_train, y[train])
    records = []
    for classifier, template in CLASSIFIERS.items():
        outcomes = {}
        for method in METHODS:
            X_aug, y_aug = variants[method]
            scores = evaluate(template, X_aug, y_aug, X_test, y[test])
            outcomes[method] = scores
            records.append({"dataset": name, "fold": number,
                            "method": method, "classifier": classifier,
                            "status": "ok", "chosen_method": "",
                            "budget": variant_info["budget"],
                            "pool_size": variant_info["pool_size"], **scores})
        selected = choices[classifier]
        records.append({"dataset": name, "fold": number,
                        "method": "ADAPTIVE", "classifier": classifier,
                        "status": "ok", "chosen_method": selected,
                        "budget": variant_info["budget"],
                        "pool_size": variant_info["pool_size"],
                        **outcomes[selected]})
    temporary = result_path.with_suffix(".csv.tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    os.replace(temporary, result_path)
    diagnostic = {"dataset": name, "fold": number,
                  "source_sha256": json.loads((DATA / f"{name}.json").read_text())["sha256_csv"],
                  "choices_saved_before_outer_test": True,
                  "decision_seconds": details["decision_seconds"],
                  "outer_budget": variant_info["budget"],
                  "outer_pool_size": variant_info["pool_size"],
                  "outer_generators": variant_info["generators"]}
    temporary = diagnostic_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(diagnostic, indent=2) + "\n")
    os.replace(temporary, diagnostic_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--outer-fold", type=int)
    args = parser.parse_args()
    for folder in ("folds", "choices", "inner", "diagnostics"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    names = tuple(SOURCES)
    if args.only:
        names = tuple(name for name in names if name in args.only)
    for name in names:
        X_raw, y, _ = load_source(name)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261125)
        for number, (train, test) in enumerate(splitter.split(X_raw, y), start=1):
            if args.outer_fold and number != args.outer_fold:
                continue
            started = time.perf_counter()
            run_fold(name, number, X_raw, y, train, test)
            print(f"{name} {number}/5: {time.perf_counter()-started:.1f}s", flush=True)
    files = [OUT / "folds" / f"{name}__fold{number}.csv"
             for name in SOURCES for number in range(1, 6)]
    if all(path.exists() for path in files):
        combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        combined.to_csv(OUT / "fold_results.csv", index=False)
        print(f"All four external sources complete: {len(combined)} rows")


if __name__ == "__main__":
    main()
