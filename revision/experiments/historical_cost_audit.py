"""Descriptive sequential timing of the actual historical BND selector."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from benchmark_pilot import build_pool
from historical_probability_benchmark import DATA, historical_indices
from keel_io import load_fold
from reference_benchmark import ROOT


OUT = ROOT / "revision/experiments/historical_cost_run"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        for fold_number in range(1, 6):
            fold = load_fold(DATA, dataset, fold_number)
            X, y = fold.X_train, fold.y_train
            budget = int((y == 0).sum() - (y == 1).sum())
            start = time.perf_counter()
            pool, _ = build_pool(X, y)
            pool_seconds = time.perf_counter() - start
            start = time.perf_counter()
            selected = historical_indices(pool, X, y, budget)
            selection_seconds = time.perf_counter() - start
            if len(selected) != budget or len(np.unique(selected)) != budget:
                raise ValueError("Historical BND index budget mismatch")
            records.append({"dataset": dataset, "fold": fold_number,
                            "pool_seconds": pool_seconds,
                            "selection_seconds": selection_seconds,
                            "total_seconds": pool_seconds + selection_seconds,
                            "pool_size": len(pool), "budget": budget})
        print(f"{dataset}: 5 folds", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "fold_timings.csv", index=False)
    summary = {"status": "sequential descriptive timing; not hardware-controlled",
               "folds": len(frame),
               "median_pool_seconds": float(frame.pool_seconds.median()),
               "median_selection_seconds": float(frame.selection_seconds.median()),
               "median_total_seconds": float(frame.total_seconds.median())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
