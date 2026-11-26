"""Equal-budget pool ablation with the actual notebook's stochastic BND rule."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from benchmark_pilot import evaluate
from historical_probability_benchmark import DATA, historical_indices
from keel_io import load_fold
from pool_overlap_benchmark import GENERATORS, POOLS, generate_parts
from reference_benchmark import CLASSIFIERS, METRICS, ROOT


OUT = ROOT / "revision/experiments/historical_pool_ablation_run"
HIST = ROOT / "revision/experiments/historical_probability_run/fold_results.csv"
SMOTE = ROOT / "revision/experiments/full_run/fold_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    for folder in (OUT / "folds", OUT / "diagnostics"):
        folder.mkdir(parents=True, exist_ok=True)
    historical = pd.read_csv(HIST)
    direct_smote = pd.read_csv(SMOTE)
    names = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    if args.only:
        names = [name for name in names if name in args.only]
    for dataset in names:
        for number in range(1, 6):
            stem = f"{dataset}__fold{number}"
            csv = OUT / "folds" / f"{stem}.csv"
            diag = OUT / "diagnostics" / f"{stem}.json"
            if csv.exists() and diag.exists():
                continue
            fold = load_fold(DATA, dataset, number)
            X, y = fold.X_train, fold.y_train
            budget = int((y == 0).sum() - (y == 1).sum())
            y_aug = np.concatenate([y, np.ones(budget, dtype=int)])
            parts, generator_report = generate_parts(X, y)
            rows = []
            pool_report = {}
            for method, members in POOLS.items():
                pool = np.vstack([parts[g] for g in members])
                if len(pool) < budget:
                    pool_report[method] = {"status": "pool_short", "size": len(pool)}
                    for classifier in CLASSIFIERS:
                        rows.append({"dataset": dataset, "fold": number,
                                     "method": method, "classifier": classifier,
                                     "status": "pool_short", "budget": budget,
                                     "pool_size": len(pool),
                                     **{metric: np.nan for metric in METRICS}})
                    continue
                indices = historical_indices(pool, X, y, budget)
                if len(indices) != budget or len(np.unique(indices)) != budget:
                    raise ValueError("Historical selector failed exact budget")
                pool_report[method] = {
                    "status": "ok", "size": len(pool),
                    "duplicate_excess": budget - len(np.unique(pool[indices], axis=0)),
                }
                for classifier, template in CLASSIFIERS.items():
                    scores = evaluate(template, np.vstack([X, pool[indices]]), y_aug,
                                      fold.X_test, fold.y_test)
                    if method == "FULL":
                        expected = historical[(historical.dataset == dataset) &
                                              (historical.fold == number) &
                                              (historical.classifier == classifier)]
                        if len(expected) != 1:
                            raise ValueError("Missing full-pool historical reference")
                        for metric in METRICS:
                            if not np.isclose(scores[metric], expected.iloc[0][metric],
                                              rtol=1e-10, atol=1e-12):
                                raise AssertionError(f"Historical reference drift {stem} {classifier} {metric}")
                    rows.append({"dataset": dataset, "fold": number,
                                 "method": method, "classifier": classifier,
                                 "status": "ok", "budget": budget,
                                 "pool_size": len(pool), **scores})
                if method in ("FULL", "ONLY_SMOTE"):
                    canonical = evaluate(
                        CLASSIFIERS["RandomForest"],
                        np.vstack([X, pool[np.sort(indices)]]), y_aug,
                        fold.X_test, fold.y_test
                    )
                    if method == "ONLY_SMOTE":
                        expected = direct_smote[
                            (direct_smote.dataset == dataset) &
                            (direct_smote.fold == number) &
                            (direct_smote.method == "SMOTE") &
                            (direct_smote.classifier == "RandomForest")
                        ]
                        if len(expected) != 1:
                            raise ValueError("Missing direct SMOTE RF reference")
                        for metric in METRICS:
                            if not np.isclose(canonical[metric], expected.iloc[0][metric],
                                              rtol=1e-10, atol=1e-12):
                                raise AssertionError(f"Canonical SMOTE RF drift {stem} {metric}")
                    rows.append({"dataset": dataset, "fold": number,
                                 "method": f"{method}_CANONICAL_RF",
                                 "classifier": "RandomForest", "status": "ok",
                                 "budget": budget, "pool_size": len(pool), **canonical})
            temporary = csv.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            os.replace(temporary, csv)
            report = {"dataset": dataset, "fold": number, "budget": budget,
                      "generator_report": generator_report,
                      "pools": pool_report,
                      "historical_full_reference_match": True,
                      "canonical_smote_RF_reference_match": True}
            temporary = diag.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            os.replace(temporary, diag)
        print(f"{dataset}: 5 folds", flush=True)
    expected = [OUT / "folds" / f"{name}__fold{number}.csv"
                for name in names for number in range(1, 6)]
    if all(path.exists() for path in expected):
        frame = pd.concat([pd.read_csv(path) for path in expected], ignore_index=True)
        if len(names) == 44:
            frame.to_csv(OUT / "fold_results.csv", index=False)
        print(f"Complete: {len(frame)} rows")


if __name__ == "__main__":
    main()
