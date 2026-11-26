"""Apply the historical stochastic selector to the existing external folds."""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from benchmark_pilot import build_pool, evaluate
from external_validation import DATA, NAMES
from historical_probability_benchmark import historical_indices
from reference_benchmark import CLASSIFIERS, METRICS, ROOT
from reference_selectors import boundary_indices


OUTPUT = ROOT / "revision/experiments/historical_external_run"
REFERENCE = ROOT / "revision/experiments/external_run/fold_results.csv"


def main() -> None:
    for directory in (OUTPUT / "folds", OUTPUT / "diagnostics"):
        directory.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE)
    for name in NAMES:
        path = DATA / f"{name}.csv"
        metadata = json.loads((DATA / f"{name}.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256_csv"]:
            raise ValueError(f"Source checksum changed: {name}")
        table = pd.read_csv(path)
        y = table.pop("__target__").to_numpy(dtype=int)
        X_raw = table.to_numpy(dtype=float)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261125)
        for number, (train, test) in enumerate(folds.split(X_raw, y), start=1):
            stem = f"{name}__fold{number}"
            csv = OUTPUT / "folds" / f"{stem}.csv"
            diag = OUTPUT / "diagnostics" / f"{stem}.json"
            if csv.exists() and diag.exists():
                continue
            started = time.perf_counter()
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(imputer.fit_transform(X_raw[train]))
            X_test = scaler.transform(imputer.transform(X_raw[test]))
            y_train, y_test = y[train], y[test]
            budget = int((y_train == 0).sum() - (y_train == 1).sum())
            pool, generators = build_pool(X_train, y_train)
            if len(pool) < budget:
                raise ValueError("Pool misses external budget")
            old_idx = boundary_indices(pool, X_train[y_train == 1], X_train[y_train == 0], budget)
            new_idx = historical_indices(pool, X_train, y_train, budget)
            y_aug = np.concatenate([y_train, np.ones(budget, dtype=int)])
            rows = []
            for classifier, template in CLASSIFIERS.items():
                expected = reference[(reference.dataset == name) &
                                     (reference.fold == number) &
                                     (reference.method == "BND") &
                                     (reference.classifier == classifier)]
                if len(expected) != 1 or expected.iloc[0].status != "ok":
                    raise ValueError("Missing saved external BND row")
                old = evaluate(template, np.vstack([X_train, pool[old_idx]]), y_aug,
                               X_test, y_test)
                for metric in METRICS:
                    if not np.isclose(old[metric], expected.iloc[0][metric],
                                      rtol=1e-10, atol=1e-12):
                        raise AssertionError(f"External reference drift: {name} {number} {classifier} {metric}")
                scores = evaluate(template, np.vstack([X_train, pool[new_idx]]), y_aug,
                                  X_test, y_test)
                rows.append({"dataset": name, "fold": number, "method": "HIST_PROB_BND",
                             "classifier": classifier, "status": "ok", "budget": budget,
                             "pool_size": len(pool), **scores})
            frame = pd.DataFrame(rows)
            temporary = csv.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            os.replace(temporary, csv)
            report = {"dataset": name, "fold": number, "budget": budget,
                      "pool_size": len(pool), "generators": generators,
                      "selected_intersection": int(len(set(old_idx) & set(new_idx))),
                      "reference_metric_match": True,
                      "seconds": time.perf_counter() - started}
            temporary = diag.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            os.replace(temporary, diag)
            print(f"{name} {number}/5: {report['seconds']:.1f}s", flush=True)
    files = [OUTPUT / "folds" / f"{name}__fold{number}.csv"
             for name in NAMES for number in range(1, 6)]
    if not all(path.exists() for path in files):
        raise ValueError("Incomplete external stochastic benchmark")
    result = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    result.to_csv(OUTPUT / "fold_results.csv", index=False)
    print(f"Complete: {len(result)} rows")


if __name__ == "__main__":
    main()
