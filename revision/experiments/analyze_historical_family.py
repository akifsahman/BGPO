"""Matched analysis of all four original notebook selector definitions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from analyze_full_run import holm_adjust
from reference_benchmark import METRICS, ROOT


OUT = ROOT / "revision/experiments/historical_family_run"
PRIMARY = ROOT / "revision/experiments/full_run/fold_results.csv"
CLASSIFIERS = ("SVM_RBF", "KNN", "RandomForest")
SELECTORS = ("HIST_RND", "HIST_DEN", "HIST_DIV", "HIST_BND")


def main() -> None:
    rows = pd.read_csv(OUT / "fold_results.csv")
    if len(rows) != 220 * 4 * 3 or (rows.status != "ok").any():
        raise ValueError("Incomplete historical family output")
    smote = pd.read_csv(PRIMARY)
    smote = smote[smote.method == "SMOTE"]
    all_rows = pd.concat([rows, smote], ignore_index=True)
    means = all_rows.groupby(["dataset", "classifier", "method"], as_index=False)[
        list(METRICS)
    ].mean()
    if len(means) != 44 * 3 * 5:
        raise ValueError("Incomplete five-fold dataset means")
    means.to_csv(OUT / "dataset_means.csv", index=False)
    paired, omnibus, descriptives = [], [], []
    for classifier in CLASSIFIERS:
        subset = means[means.classifier == classifier]
        for metric in METRICS:
            panel = subset.pivot(index="dataset", columns="method", values=metric)
            if panel.isna().any().any():
                raise ValueError("Missing paired dataset metric")
            ranks = panel[list(SELECTORS)].rank(axis=1, ascending=False, method="average")
            friedman_p = float(friedmanchisquare(*[panel[m] for m in SELECTORS]).pvalue)
            omnibus.append({"classifier": classifier, "metric": metric,
                            "friedman_p_raw": friedman_p})
            for method in (*SELECTORS, "SMOTE"):
                descriptives.append({"classifier": classifier, "metric": metric,
                                     "method": method, "mean": float(panel[method].mean()),
                                     "mean_rank_among_four": float(ranks[method].mean())
                                     if method in SELECTORS else np.nan})
            for comparator in ("HIST_RND", "HIST_DEN", "HIST_DIV", "SMOTE"):
                delta = panel.HIST_BND - panel[comparator]
                p = 1.0 if np.all(np.isclose(delta, 0)) else float(wilcoxon(
                    delta, zero_method="pratt", alternative="two-sided", method="auto"
                ).pvalue)
                paired.append({"classifier": classifier, "metric": metric,
                               "comparator": comparator,
                               "BND_mean": float(panel.HIST_BND.mean()),
                               "comparator_mean": float(panel[comparator].mean()),
                               "mean_difference": float(delta.mean()),
                               "wins": int((delta > 1e-12).sum()),
                               "ties": int((delta.abs() <= 1e-12).sum()),
                               "losses": int((delta < -1e-12).sum()),
                               "p_raw": p})
    paired = pd.DataFrame(paired)
    paired["p_holm_all60"] = holm_adjust(paired.p_raw.tolist())
    paired.to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(omnibus).to_csv(OUT / "friedman_tests.csv", index=False)
    pd.DataFrame(descriptives).to_csv(OUT / "descriptive_summary.csv", index=False)
    summary = {
        "datasets": 44, "folds": 220, "classifier_rows": len(rows),
        "Holm60_significant": paired.loc[paired.p_holm_all60 < 0.05,
            ["classifier", "metric", "comparator", "mean_difference", "p_holm_all60"]
        ].to_dict(orient="records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(descriptives).query("metric in ['F1','BalancedAccuracy','GMean']").to_string(index=False))
    print(paired[paired.metric.isin(["F1", "BalancedAccuracy", "GMean"])][
        ["classifier", "metric", "comparator", "mean_difference", "p_holm_all60"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
