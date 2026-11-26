"""Compact, provenance-aware summary of the previously viewed external sources."""

from __future__ import annotations

import json

import pandas as pd

from reference_benchmark import ROOT


SOURCE = ROOT / "revision/experiments/historical_external_run/dataset_differences.csv"
OUTPUT = ROOT / "revision/experiments/historical_external_run"


def main() -> None:
    comparisons = pd.read_csv(SOURCE)
    smote = comparisons[comparisons.comparator == "SMOTE"].copy()
    if (len(smote) != 3 * 3 * 5 or
            smote.duplicated(["dataset", "classifier", "metric"]).any()):
        raise ValueError("Incomplete external SMOTE comparison")
    short = smote[smote.metric.isin(("F1", "BalancedAccuracy", "GMean"))][
        ["dataset", "classifier", "metric", "historical", "comparator_value",
         "difference"]
    ].sort_values(["dataset", "classifier", "metric"])
    short.to_csv(OUTPUT / "SMOTE_highlight_table.csv", index=False)
    primary = smote[smote.metric == "GMean"]
    summary = {
        "status": "exploratory: these three sources were already viewed before this summary",
        "source_count": 3, "source_classifier_blocks": len(primary),
        "GMean_better_blocks": int((primary.difference > 1e-12).sum()),
        "GMean_worse_blocks": int((primary.difference < -1e-12).sum()),
        "GMean_mean_difference_over_nine_blocks": float(primary.difference.mean()),
    }
    (OUTPUT / "SMOTE_highlight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(primary[["dataset", "classifier", "historical", "comparator_value",
                   "difference"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
