"""Descriptive, post-hoc external comparison of historical selection."""

from __future__ import annotations

import json

import pandas as pd

from reference_benchmark import METRICS, ROOT


OUT = ROOT / "revision/experiments/historical_external_run"
REFERENCE = ROOT / "revision/experiments/external_run/fold_results.csv"


def main() -> None:
    historical = pd.read_csv(OUT / "fold_results.csv")
    reference = pd.read_csv(REFERENCE)
    all_rows = pd.concat([historical, reference], ignore_index=True)
    if len(historical) != 45 or len(all_rows) != 270 or (all_rows.status != "ok").any():
        raise ValueError("Incomplete external comparison")
    means = all_rows.groupby(["dataset", "classifier", "method"], as_index=False)[
        list(METRICS)
    ].mean()
    means.to_csv(OUT / "dataset_means.csv", index=False)
    rows = []
    for (dataset, classifier), block in means.groupby(["dataset", "classifier"]):
        for metric in METRICS:
            lookup = dict(zip(block.method, block[metric]))
            for comparator in ("BND", "DIV", "SMOTE", "BND_QUOTA"):
                rows.append({"dataset": dataset, "classifier": classifier,
                             "metric": metric, "comparator": comparator,
                             "historical": float(lookup["HIST_PROB_BND"]),
                             "comparator_value": float(lookup[comparator]),
                             "difference": float(lookup["HIST_PROB_BND"] - lookup[comparator])})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "dataset_differences.csv", index=False)
    gmean = comparison[comparison.metric == "GMean"]
    summary = {comparator: {
        "better_blocks": int((group.difference > 1e-12).sum()),
        "tie_blocks": int((group.difference.abs() <= 1e-12).sum()),
        "worse_blocks": int((group.difference < -1e-12).sum()),
        "mean_difference_over_nine_blocks": float(group.difference.mean()),
    } for comparator, group in gmean.groupby("comparator")}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(gmean[["dataset", "classifier", "comparator", "historical",
                 "comparator_value", "difference"]].to_string(index=False))


if __name__ == "__main__":
    main()
