"""Within-run, paired generation-time audit of historical BND and SMOTE.

No classifier fitting is timed. Methods alternate order across three repetitions
per fold. This is descriptive timing on one machine, not a portable benchmark.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from benchmark_pilot import build_pool
from historical_probability_benchmark import DATA, historical_indices
from keel_io import load_fold
from reference_benchmark import ROOT, classic_variant

OUT = ROOT / "revision/experiments/historical_cost_paired_run"


def one(X: np.ndarray, y: np.ndarray, budget: int, method: str) -> tuple[float, int]:
    start = time.perf_counter()
    if method == "SMOTE":
        variant, report = classic_variant(X, y, "SMOTE", budget)
        if variant is None or report["status"] != "ok":
            raise ValueError(f"SMOTE generation failed: {report}")
        added = len(variant[0]) - len(X)
    else:
        pool, _ = build_pool(X, y)
        selected = historical_indices(pool, X, y, budget)
        if len(np.unique(selected)) != budget:
            raise ValueError("Historical BND failed index budget")
        added = len(selected)
    elapsed = time.perf_counter() - start
    if added != budget:
        raise ValueError(f"{method} missed budget: {added} != {budget}")
    return elapsed, added


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        for fold_number in range(1, 6):
            fold = load_fold(DATA, dataset, fold_number)
            X, y = fold.X_train, fold.y_train
            budget = int((y == 0).sum() - (y == 1).sum())
            for repeat in range(1, 4):
                order = ("SMOTE", "HIST_BND") if (fold_number + repeat) % 2 else ("HIST_BND", "SMOTE")
                for position, method in enumerate(order, 1):
                    seconds, added = one(X, y, budget, method)
                    rows.append({"dataset": dataset, "fold": fold_number,
                                 "repeat": repeat, "position": position,
                                 "method": method, "seconds": seconds,
                                 "budget": budget, "added": added})
        print(f"{dataset}: paired 5 folds x 3 repeats", flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 44 * 5 * 3 * 2:
        raise ValueError("Incomplete paired timing")
    frame.to_csv(OUT / "timings.csv", index=False)
    fold_median = frame.groupby(["dataset", "fold", "method"]).seconds.median().unstack()
    fold_median["ratio"] = fold_median.HIST_BND / fold_median.SMOTE
    fold_median["difference"] = fold_median.HIST_BND - fold_median.SMOTE
    fold_median.reset_index().to_csv(OUT / "fold_medians.csv", index=False)
    summary = {"design": "same-process paired wall time; 220 folds, 3 repeats, alternating order; generation and selection only",
               "n_folds": len(fold_median), "n_calls": len(frame),
               "median_smote_seconds": float(fold_median.SMOTE.median()),
               "median_historical_BND_seconds": float(fold_median.HIST_BND.median()),
               "median_paired_difference_seconds": float(fold_median.difference.median()),
               "median_paired_ratio": float(fold_median.ratio.median()),
               "BND_slower_folds": int((fold_median.HIST_BND > fold_median.SMOTE).sum()),
               "BND_faster_folds": int((fold_median.HIST_BND < fold_median.SMOTE).sum()),
               "limitations": "Single machine and current Python environment; no classifier fit or memory measurement; results are descriptive."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
