"""Paired exploratory analysis of historical BND generator-pool ablation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_full_run import holm_adjust
from pool_overlap_benchmark import POOLS
from reference_benchmark import METRICS, ROOT


OUT = ROOT / "revision/experiments/historical_pool_ablation_run"
COMPARATORS = tuple(method for method in POOLS if method != "FULL")


def paired(a: pd.Series, b: pd.Series) -> tuple[float, float, int, int, int]:
    diff = a - b
    p = (1.0 if np.all(np.isclose(diff, 0)) else
         float(wilcoxon(diff, zero_method="pratt", alternative="two-sided",
                        method="auto").pvalue))
    return (float(diff.mean()), p, int((diff > 1e-12).sum()),
            int((diff.abs() <= 1e-12).sum()), int((diff < -1e-12).sum()))


def main() -> None:
    rows = pd.read_csv(OUT / "fold_results.csv")
    if rows.duplicated(["dataset", "fold", "classifier", "method"]).any():
        raise ValueError("Duplicate fold result")
    complete = rows.groupby(["dataset", "classifier", "method"]).agg(
        n_folds=("fold", "nunique"), n_ok=("status", lambda x: (x == "ok").sum())
    ).reset_index()
    complete = complete[(complete.n_folds == 5) & (complete.n_ok == 5)]
    means = rows.merge(complete[["dataset", "classifier", "method"]],
                       on=["dataset", "classifier", "method"]).groupby(
                           ["dataset", "classifier", "method"], as_index=False
                       )[list(METRICS)].mean()
    means.to_csv(OUT / "complete_dataset_means.csv", index=False)
    comparisons = []
    for classifier in ("SVM_RBF", "KNN", "RandomForest"):
        for metric in METRICS:
            panel = means[means.classifier == classifier].pivot(
                index="dataset", columns="method", values=metric
            )
            for comparator in COMPARATORS:
                pair = panel[["FULL", comparator]].dropna()
                if len(pair) < 2:
                    raise ValueError("Too few complete paired datasets")
                difference, p, wins, ties, losses = paired(pair.FULL, pair[comparator])
                comparisons.append({
                    "classifier": classifier, "metric": metric,
                    "comparator": comparator, "n_datasets": len(pair),
                    "full_mean_on_pair": float(pair.FULL.mean()),
                    "comparator_mean_on_pair": float(pair[comparator].mean()),
                    "full_minus_comparator": difference,
                    "wins": wins, "ties": ties, "losses": losses, "p_raw": p,
                })
    comparisons = pd.DataFrame(comparisons)
    comparisons["p_holm_all120"] = holm_adjust(comparisons.p_raw.tolist())
    primary = (comparisons.metric == "GMean") & (
        comparisons.comparator == "ONLY_SMOTE"
    )
    comparisons["p_holm_primary3"] = np.nan
    comparisons.loc[primary, "p_holm_primary3"] = holm_adjust(
        comparisons.loc[primary, "p_raw"].tolist()
    )
    comparisons.to_csv(OUT / "paired_pool_comparisons.csv", index=False)
    rf = means[means.classifier == "RandomForest"].pivot(
        index="dataset", columns="method", values="GMean"
    )
    canonical = []
    for full, comparator in (("FULL", "ONLY_SMOTE"),
                             ("FULL_CANONICAL_RF", "ONLY_SMOTE_CANONICAL_RF")):
        pair = rf[[full, comparator]].dropna()
        difference, p, wins, ties, losses = paired(pair[full], pair[comparator])
        canonical.append({"order": "canonical" if "CANONICAL" in full else "draw",
                          "n_datasets": len(pair), "full_minus_smote_only_GMean": difference,
                          "wins": wins, "ties": ties, "losses": losses, "p_raw": p})
    canonical_frame = pd.DataFrame(canonical)
    canonical_frame.to_csv(OUT / "RF_row_order_sensitivity.csv", index=False)
    summary = {
        "status": "exploratory after earlier KEEL outcomes viewed",
        "full_reference_complete": bool(
            len(rows[(rows.method == "FULL") & (rows.status == "ok")]) == 660
        ),
        "primary_full_minus_smote_only_GMean": comparisons.loc[primary].to_dict(
            orient="records"
        ),
        "RF_order_sensitivity": canonical,
        "Holm120_significant": comparisons.loc[
            comparisons.p_holm_all120 < 0.05,
            ["classifier", "metric", "comparator", "n_datasets",
             "full_minus_comparator", "p_holm_all120"]
        ].to_dict(orient="records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(comparisons[comparisons.metric == "GMean"][
        ["classifier", "comparator", "n_datasets", "full_mean_on_pair",
         "comparator_mean_on_pair", "full_minus_comparator", "p_holm_all120",
         "p_holm_primary3"]
    ].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
