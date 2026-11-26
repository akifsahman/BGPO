"""Build publication-ready source tables for the actual notebook selector.

This is an exploratory revision after seeing the submitted results. It keeps
the corrected benchmark's folds, scaler, generator pool, budget and templates.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_full_run import holm_adjust
from reference_benchmark import METRICS, ROOT


OUT = ROOT / "revision/experiments/historical_primary_tables"
HIST = ROOT / "revision/experiments/historical_family_run/fold_results.csv"
CLASSIC = ROOT / "revision/experiments/full_run/fold_results.csv"
METHODS = (
    "Original", "ROS", "SMOTE", "ADASYN", "Borderline_SMOTE1",
    "Borderline_SMOTE2", "MWMOTE", "G_SMOTE", "kmeans_SMOTE",
)
SELECTORS = ("HIST_RND", "HIST_DIV", "HIST_DEN", "HIST_BND")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(HIST)
    classic = pd.read_csv(CLASSIC)
    classic = classic[classic.method.isin(METHODS)]
    rows = pd.concat([hist, classic], ignore_index=True)
    if (len(hist) != 220 * 4 * 3 or
            rows.duplicated(["dataset", "fold", "method", "classifier"]).any()):
        raise ValueError("Incomplete or duplicated historical selector data")
    complete = rows.groupby(["dataset", "classifier", "method"]).agg(
        n_folds=("fold", "nunique"), n_ok=("status", lambda s: (s == "ok").sum())
    ).reset_index()
    complete = complete[(complete.n_folds == 5) & (complete.n_ok == 5)]
    means = rows.merge(
        complete[["dataset", "classifier", "method"]],
        on=["dataset", "classifier", "method"], how="inner"
    ).groupby(["dataset", "classifier", "method"], as_index=False)[list(METRICS)].mean()
    means.to_csv(OUT / "complete_dataset_means.csv", index=False)
    descriptive = means.groupby(["classifier", "method"]).agg(
        n_datasets=("dataset", "nunique"),
        **{metric: (metric, "mean") for metric in METRICS}
    ).reset_index()
    descriptive.to_csv(OUT / "descriptive_table.csv", index=False)
    comparisons = []
    for classifier in ("SVM_RBF", "KNN", "RandomForest"):
        subset = means[means.classifier == classifier]
        for metric in METRICS:
            panel = subset.pivot(index="dataset", columns="method", values=metric)
            for comparator in METHODS:
                pair = panel[["HIST_BND", comparator]].dropna()
                if len(pair) < 2:
                    raise ValueError("Insufficient complete datasets")
                diff = pair.HIST_BND - pair[comparator]
                p = (1.0 if np.all(np.isclose(diff, 0)) else
                     float(wilcoxon(diff, zero_method="pratt",
                                    alternative="two-sided", method="auto").pvalue))
                comparisons.append({
                    "classifier": classifier, "metric": metric,
                    "comparator": comparator, "n_datasets": len(pair),
                    "BND_mean_on_pair": float(pair.HIST_BND.mean()),
                    "comparator_mean_on_pair": float(pair[comparator].mean()),
                    "mean_difference": float(diff.mean()),
                    "wins": int((diff > 1e-12).sum()),
                    "ties": int((diff.abs() <= 1e-12).sum()),
                    "losses": int((diff < -1e-12).sum()),
                    "p_raw": p,
                })
    comparisons = pd.DataFrame(comparisons)
    comparisons["p_holm_within_classifier_metric"] = np.nan
    for _, indices in comparisons.groupby(["classifier", "metric"]).groups.items():
        comparisons.loc[indices, "p_holm_within_classifier_metric"] = holm_adjust(
            comparisons.loc[indices, "p_raw"].tolist()
        )
    comparisons["p_holm_all135"] = holm_adjust(comparisons.p_raw.tolist())
    comparisons.to_csv(OUT / "classical_paired_comparisons.csv", index=False)
    summary = {
        "status": "exploratory historical-primary manuscript preparation",
        "dataset_count": 44, "fold_count": 220,
        "historical_selector_rows": len(hist),
        "all_historical_selectors_complete": bool((hist.status == "ok").all()),
        "significant_classical_differences": comparisons.loc[
            (comparisons.p_holm_all135 < 0.05) &
            (comparisons.comparator != "Original"),
            ["classifier", "metric", "comparator", "n_datasets",
             "mean_difference", "p_holm_all135"]
        ].to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(descriptive[descriptive.method.isin((*SELECTORS, "Original", "SMOTE"))][
        ["classifier", "method", "n_datasets", "F1", "BalancedAccuracy", "GMean"]
    ].to_string(index=False))
    print(comparisons[
        (comparisons.comparator.isin(("Original", "SMOTE"))) &
        (comparisons.metric.isin(("F1", "BalancedAccuracy", "GMean")))
    ][["classifier", "metric", "comparator", "n_datasets",
       "mean_difference", "wins", "ties", "losses",
       "p_holm_within_classifier_metric"]].to_string(index=False))


if __name__ == "__main__":
    main()
