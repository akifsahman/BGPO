"""Audit train-only choices and summarize the four untouched external sources."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from adaptive_external_benchmark import METHODS, OUT
from fetch_adaptive_external import SOURCES
from reference_benchmark import CLASSIFIERS, METRICS


def main() -> None:
    results = pd.read_csv(OUT / "fold_results.csv")
    expected = len(SOURCES) * 5 * len(CLASSIFIERS) * 4
    if (len(results) != expected or (results.status != "ok").any() or
            results.duplicated(["dataset", "fold", "method", "classifier"]).any()):
        raise ValueError("Incomplete adaptive external results")
    audit = []
    for source in SOURCES:
        for fold in range(1, 6):
            stem = f"{source}__fold{fold}"
            choice_path = OUT / "choices" / f"{stem}.json"
            result_path = OUT / "folds" / f"{stem}.csv"
            inner_path = OUT / "inner" / f"{stem}.csv"
            detail = json.loads(choice_path.read_text())
            if (choice_path.stat().st_mtime > result_path.stat().st_mtime or
                    inner_path.stat().st_mtime > result_path.stat().st_mtime):
                raise ValueError("Inner decision saved after outer result")
            inner = pd.read_csv(inner_path)
            for classifier in CLASSIFIERS:
                subset = inner[inner.classifier == classifier]
                if (len(subset) != 9 or
                        set(subset.groupby("method").inner_fold.nunique().to_dict()) != set(METHODS)):
                    raise ValueError("Incomplete inner fold audit")
                means = subset.groupby("method").GMean.mean()
                expected_choice = METHODS[int(np.argmax([means[m] for m in METHODS]))]
                if detail["choices"][classifier] != expected_choice:
                    raise ValueError("Saved choice differs from inner G-Mean rule")
                panel = results[(results.dataset == source) &
                                (results.fold == fold) &
                                (results.classifier == classifier)]
                adaptive = panel[panel.method == "ADAPTIVE"].iloc[0]
                chosen = panel[panel.method == expected_choice].iloc[0]
                if adaptive.chosen_method != expected_choice:
                    raise ValueError("Adaptive row has wrong chosen method")
                for metric in METRICS:
                    if not np.isclose(adaptive[metric], chosen[metric], atol=1e-12):
                        raise ValueError("Adaptive row does not match chosen method")
                audit.append({"dataset": source, "fold": fold,
                              "classifier": classifier, "chosen_method": expected_choice,
                              "inner_best_GMean": float(means[expected_choice]),
                              "inner_SMOTE_GMean": float(means["SMOTE"]),
                              "decision_seconds": detail["decision_seconds"]})
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(OUT / "selection_audit.csv", index=False)
    means = results.groupby(["dataset", "classifier", "method"], as_index=False)[
        list(METRICS)
    ].mean()
    means.to_csv(OUT / "dataset_means.csv", index=False)
    contrasts = []
    for dataset in SOURCES:
        for classifier in CLASSIFIERS:
            panel = means[(means.dataset == dataset) &
                          (means.classifier == classifier)].set_index("method")
            for metric in METRICS:
                for comparator in METHODS:
                    contrasts.append({
                        "dataset": dataset, "classifier": classifier,
                        "metric": metric, "comparator": comparator,
                        "adaptive": float(panel.loc["ADAPTIVE", metric]),
                        "comparator_value": float(panel.loc[comparator, metric]),
                        "difference": float(panel.loc["ADAPTIVE", metric] -
                                            panel.loc[comparator, metric]),
                    })
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(OUT / "dataset_differences.csv", index=False)
    primary = contrast_frame[(contrast_frame.metric == "GMean") &
                             (contrast_frame.comparator == "SMOTE")]
    summary = {
        "status": "four metadata-fixed, previously unscored sources; descriptive transfer check",
        "sources": len(SOURCES), "outer_folds": len(SOURCES) * 5,
        "adaptive_classifier_folds": len(audit_frame),
        "choices": audit_frame.chosen_method.value_counts().to_dict(),
        "median_inner_decision_seconds_per_outer_fold": float(
            audit_frame.drop_duplicates(["dataset", "fold"]).decision_seconds.median()
        ),
        "primary_GMean_vs_SMOTE_by_source_classifier": primary.to_dict(orient="records"),
        "primary_mean_difference_over_12_blocks": float(primary.difference.mean()),
        "primary_wins": int((primary.difference > 1e-12).sum()),
        "primary_ties": int((primary.difference.abs() <= 1e-12).sum()),
        "primary_losses": int((primary.difference < -1e-12).sum()),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(primary.to_string(index=False))
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "primary_GMean_vs_SMOTE_by_source_classifier"}, indent=2))


if __name__ == "__main__":
    main()
