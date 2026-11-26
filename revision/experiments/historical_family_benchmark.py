"""Re-evaluate all four selectors from the historical notebook on fixed folds.

This preserves the original notebook selector definitions while using the
corrected benchmark's paired, train-only preprocessing and fixed four-generator
pool. It is exploratory because the historical submitted outcomes are known.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_pilot import build_pool, evaluate
from keel_io import load_fold
from reference_benchmark import CLASSIFIERS, METRICS, ROOT


sys.path.insert(0, str(ROOT / "historical/notebook"))
from benchmark_pipeline_pool_ablation import (  # noqa: E402
    build_neighbor_context,
    proposed_boundary_from_pool,
    proposed_density_from_pool,
    proposed_diversity_from_pool,
    proposed_random_from_pool,
)


DATA = ROOT / "Pool-Select/datasets_keel"
REFERENCE = ROOT / "revision/experiments/historical_probability_run/fold_results.csv"
OUT = ROOT / "revision/experiments/historical_family_run"
METHODS = {
    "HIST_RND": proposed_random_from_pool,
    "HIST_DEN": proposed_density_from_pool,
    "HIST_DIV": proposed_diversity_from_pool,
    "HIST_BND": proposed_boundary_from_pool,
}


def main() -> None:
    for folder in (OUT / "folds", OUT / "diagnostics"):
        folder.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE)
    for dataset in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        for number in range(1, 6):
            stem = f"{dataset}__fold{number}"
            csv = OUT / "folds" / f"{stem}.csv"
            diag = OUT / "diagnostics" / f"{stem}.json"
            if csv.exists() and diag.exists():
                continue
            fold = load_fold(DATA, dataset, number)
            X, y = fold.X_train, fold.y_train
            budget = int((y == 0).sum() - (y == 1).sum())
            pool, _ = build_pool(X, y)
            pool_y = np.ones(len(pool), dtype=int)
            context = build_neighbor_context(X, y)
            rows = []
            duplicate_excess = {}
            for method, selector in METHODS.items():
                if method in ("HIST_DEN", "HIST_BND"):
                    X_aug, y_aug = selector(X, y, budget, pool, pool_y, context)
                else:
                    X_aug, y_aug = selector(X, y, budget, pool, pool_y)
                if len(X_aug) != len(X) + budget or int((y_aug == 1).sum()) != int((y == 0).sum()):
                    raise ValueError(f"Historical {method} missed budget: {dataset} {number}")
                duplicate_excess[method] = budget - len(np.unique(X_aug[len(X):], axis=0))
                for classifier, template in CLASSIFIERS.items():
                    scores = evaluate(template, X_aug, y_aug, fold.X_test, fold.y_test)
                    if method == "HIST_BND":
                        expected = reference[(reference.dataset == dataset) &
                                             (reference.fold == number) &
                                             (reference.classifier == classifier)]
                        if len(expected) != 1:
                            raise ValueError("Missing stochastic BND reference")
                        for metric in METRICS:
                            if not np.isclose(scores[metric], expected.iloc[0][metric],
                                              rtol=1e-10, atol=1e-12):
                                raise AssertionError(f"Historical reference drift {dataset} {number} {classifier} {metric}")
                    rows.append({"dataset": dataset, "fold": number, "method": method,
                                 "classifier": classifier, "status": "ok", "budget": budget,
                                 "pool_size": len(pool), **scores})
            temporary = csv.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            os.replace(temporary, csv)
            temporary = diag.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"dataset": dataset, "fold": number,
                                             "budget": budget, "pool_size": len(pool),
                                             "duplicate_excess": duplicate_excess,
                                             "historical_BND_reference_match": True}, indent=2) + "\n")
            os.replace(temporary, diag)
        print(f"{dataset}: 5 folds", flush=True)
    files = sorted((OUT / "folds").glob("*.csv"))
    if len(files) != 220:
        raise ValueError("Incomplete historical selector family")
    result = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if len(result) != 220 * 4 * 3:
        raise ValueError("Unexpected historical family row count")
    result.to_csv(OUT / "fold_results.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "status": "post-hoc historical selector-family audit",
        "selector_source": "historical/notebook/benchmark_pipeline_pool_ablation.py",
        "unchanged": "corrected 44-KEEL folds, training-only preprocessing, four-generator pool, budget, classifiers, metrics",
        "historical_BND_reference_verified_per_fold": True,
    }, indent=2) + "\n")
    print(f"Complete: {len(result)} rows")


if __name__ == "__main__":
    main()
