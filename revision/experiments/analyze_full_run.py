"""Analyze completed five-fold dataset means from the reference benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "revision/experiments/full_run"
METRICS = ("F1", "BalancedAccuracy", "GMean", "ROC_AUC", "MCC")
CLASSIFIERS = ("SVM_RBF", "KNN", "RandomForest")
SELECTION = ("RND", "DIV", "DEN", "BND")
CLASSICAL = (
    "Original", "ROS", "SMOTE", "ADASYN", "Borderline_SMOTE1",
    "Borderline_SMOTE2", "MWMOTE", "G_SMOTE", "kmeans_SMOTE", "BND",
)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted.tolist()


def complete_dataset_means(fold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = ["dataset", "method", "classifier"]
    valid = fold[fold["status"] == "ok"].copy()
    counts = valid.groupby(group)["fold"].nunique().rename("n_folds")
    means = valid.groupby(group)[list(METRICS)].mean()
    all_means = means.join(counts).reset_index()
    complete = all_means[all_means["n_folds"] == 5].copy()
    return all_means, complete


def matrix(complete: pd.DataFrame, classifier: str, metric: str, methods: tuple[str, ...]) -> pd.DataFrame:
    panel = complete[
        (complete["classifier"] == classifier)
        & (complete["method"].isin(methods))
    ].pivot(index="dataset", columns="method", values=metric)
    return panel.reindex(columns=list(methods)).dropna()


def friedman_rows(complete: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    ranks = []
    for family, methods in (("selection", SELECTION), ("classical", CLASSICAL)):
        for classifier in CLASSIFIERS:
            for metric in METRICS:
                panel = matrix(complete, classifier, metric, methods)
                n = len(panel)
                if n >= 3:
                    stat, p = friedmanchisquare(*(panel[method] for method in methods))
                    rank_values = np.vstack([
                        rankdata(-row, method="average") for row in panel.to_numpy()
                    ])
                    mean_ranks = rank_values.mean(axis=0)
                else:
                    stat = p = np.nan
                    mean_ranks = np.full(len(methods), np.nan)
                results.append({
                    "family": family, "classifier": classifier, "metric": metric,
                    "n_complete_datasets": n, "statistic": stat, "p_value": p,
                })
                for method, rank in zip(methods, mean_ranks):
                    ranks.append({
                        "family": family, "classifier": classifier, "metric": metric,
                        "method": method, "n_complete_datasets": n,
                        "mean_rank": rank,
                    })
    return pd.DataFrame(results), pd.DataFrame(ranks)


def paired_rows(complete: pd.DataFrame) -> pd.DataFrame:
    results = []
    for family, methods in (("selection", SELECTION), ("classical", CLASSICAL)):
        comparators = [method for method in methods if method != "BND"]
        for classifier in CLASSIFIERS:
            for metric in METRICS:
                block = []
                for comparator in comparators:
                    panel = matrix(complete, classifier, metric, ("BND", comparator))
                    differences = panel["BND"] - panel[comparator]
                    if len(panel) < 3:
                        p = np.nan
                    elif np.all(np.isclose(differences, 0)):
                        p = 1.0
                    else:
                        p = wilcoxon(
                            differences,
                            zero_method="pratt",
                            alternative="two-sided",
                            method="auto",
                        ).pvalue
                    block.append({
                        "family": family,
                        "classifier": classifier,
                        "metric": metric,
                        "comparator": comparator,
                        "n_complete_datasets": len(panel),
                        "mean_BND": panel["BND"].mean() if len(panel) else np.nan,
                        "mean_comparator": panel[comparator].mean() if len(panel) else np.nan,
                        "mean_difference": differences.mean() if len(panel) else np.nan,
                        "median_difference": differences.median() if len(panel) else np.nan,
                        "BND_wins": int((differences > 0).sum()),
                        "ties": int((differences == 0).sum()),
                        "BND_losses": int((differences < 0).sum()),
                        "p_raw": p,
                    })
                p_values = [item["p_raw"] for item in block]
                valid_indices = [i for i, p in enumerate(p_values) if np.isfinite(p)]
                if valid_indices:
                    adjusted = holm_adjust([p_values[i] for i in valid_indices])
                    for index, p_holm in zip(valid_indices, adjusted):
                        block[index]["p_holm"] = p_holm
                        block[index]["interpretation"] = (
                            "BND higher" if p_holm < 0.05 and block[index]["mean_difference"] > 0
                            else "BND lower" if p_holm < 0.05 and block[index]["mean_difference"] < 0
                            else "not significant"
                        )
                results.extend(block)
    return pd.DataFrame(results)


def descriptive_rows(complete: pd.DataFrame) -> pd.DataFrame:
    melted = complete.melt(
        id_vars=["dataset", "method", "classifier", "n_folds"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="score",
    )
    return melted.groupby(["method", "classifier", "metric"]).agg(
        n_complete_datasets=("dataset", "nunique"),
        mean=("score", "mean"),
        sd=("score", "std"),
        median=("score", "median"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run
    fold = pd.read_csv(run / "fold_results.csv")
    expected = 44 * 5 * 13 * 3
    if len(fold) != expected or fold[["dataset", "fold", "method", "classifier"]].duplicated().any():
        raise ValueError(f"Expected {expected} unique fold records; found {len(fold)}")
    if fold["dataset"].nunique() != 44 or fold["fold"].nunique() != 5:
        raise ValueError("Expected 44 datasets and 5 folds")
    all_means, complete = complete_dataset_means(fold)
    descriptive = descriptive_rows(complete)
    friedman, ranks = friedman_rows(complete)
    paired = paired_rows(complete)
    all_means.to_csv(run / "dataset_means_with_fold_counts.csv", index=False)
    complete.to_csv(run / "dataset_means_complete.csv", index=False)
    descriptive.to_csv(run / "descriptive_summary.csv", index=False)
    friedman.to_csv(run / "friedman_tests.csv", index=False)
    ranks.to_csv(run / "mean_ranks_common_sets.csv", index=False)
    paired.to_csv(run / "paired_wilcoxon_holm.csv", index=False)
    failure = fold[fold["status"] != "ok"].drop_duplicates(
        ["dataset", "fold", "method", "status"]
    ).groupby(["method", "status"]).agg(
        affected_folds=("fold", "size"),
        affected_datasets=("dataset", "nunique"),
    ).reset_index()
    failure.to_csv(run / "unavailable_methods.csv", index=False)
    summary = {
        "fold_rows": len(fold),
        "successful_rows": int((fold["status"] == "ok").sum()),
        "complete_dataset_method_classifier_rows": len(complete),
        "selection_common_datasets": int(friedman[friedman.family == "selection"]["n_complete_datasets"].min()),
        "classical_common_datasets": int(friedman[friedman.family == "classical"]["n_complete_datasets"].min()),
        "unavailable_method_counts": failure.to_dict(orient="records"),
    }
    (run / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
