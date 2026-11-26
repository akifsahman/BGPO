"""Generate historical-primary LaTeX tables solely from saved paired outputs."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tables"
EXP = ROOT / "revision/experiments"
METRICS = ("F1", "BalancedAccuracy", "GMean", "ROC_AUC", "MCC")
CLFS = ("SVM_RBF", "KNN", "RandomForest")
CLF_LABEL = {"SVM_RBF": "SVM (RBF)", "KNN": "KNN", "RandomForest": "RF"}
SELECTOR_LABEL = {
    "HIST_RND": "RND", "HIST_DIV": "DIV",
    "HIST_DEN": "DEN", "HIST_BND": "BND",
}
CLASSIC_LABEL = {
    "Original": "Original", "ROS": "ROS", "SMOTE": "SMOTE",
    "ADASYN": "ADASYN", "Borderline_SMOTE1": "B-SMOTE1",
    "Borderline_SMOTE2": "B-SMOTE2", "MWMOTE": "MWMOTE",
    "G_SMOTE": "G-SMOTE", "kmeans_SMOTE": "K-Means SMOTE",
    "HIST_BND": "BND",
}
CLASSIC_ORDER = (
    "Original", "ROS", "SMOTE", "ADASYN", "Borderline_SMOTE1",
    "Borderline_SMOTE2", "MWMOTE", "G_SMOTE", "kmeans_SMOTE",
    "HIST_BND",
)


def f3(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def p3(value: float) -> str:
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def name_tex(name: str) -> str:
    return name.replace("_", r"\_").replace("-", r"-\allowbreak{}")


def table(caption: str, label: str, header: list[str],
          rows: list[list[str]], columns: str, size: str = r"\small") -> str:
    lines = [
        r"\begingroup" + size + r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{longtable}}{{@{{}}{columns}@{{}}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines += [r"\end{longtable}", r"\endgroup", ""]
    return "\n".join(lines)


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content)


def external_panel() -> pd.DataFrame:
    old = pd.read_csv(EXP / "historical_external_run/dataset_differences.csv")
    old = old[(old.metric == "GMean") & (old.comparator == "SMOTE")]
    old = old[["dataset", "classifier", "historical", "comparator_value", "difference"]].copy()
    old["provenance"] = "Earlier external"
    new = pd.read_csv(EXP / "adaptive_external_run/dataset_means.csv")
    new = new[new.method.isin(("HIST_BND", "SMOTE"))]
    pivot = new.pivot(index=["dataset", "classifier"], columns="method", values="GMean").reset_index()
    pivot["difference"] = pivot.HIST_BND - pivot.SMOTE
    pivot = pivot.rename(columns={"HIST_BND": "historical", "SMOTE": "comparator_value"})
    pivot["provenance"] = "New external"
    panel = pd.concat([old, pivot], ignore_index=True)
    if len(panel) != 21 or panel.duplicated(["dataset", "classifier"]).any():
        raise ValueError("Incomplete seven-source panel")
    panel.to_csv(OUT / "external_BND_vs_SMOTE.csv", index=False)
    return panel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(EXP / "historical_family_run/descriptive_summary.csv")
    rows = []
    for clf in CLFS:
        for metric in METRICS:
            cells = []
            for method in SELECTOR_LABEL:
                match = desc[(desc.classifier == clf) & (desc.metric == metric) &
                             (desc.method == method)]
                if len(match) != 1:
                    raise ValueError("Missing selector descriptive value")
                record = match.iloc[0]
                cells.append(f"{record['mean']:.3f} ({record['mean_rank_among_four']:.2f})")
            metric_label = {"BalancedAccuracy": "BAcc", "GMean": "G-Mean",
                            "ROC_AUC": "ROC-AUC"}.get(metric, metric)
            rows.append([CLF_LABEL[clf], metric_label, *cells])
    write("table3.tex", table(
        "Historical notebook selection rules on 44 KEEL datasets: five-fold dataset means (mean rank)",
        "tab:3", ["Classifier", "Metric", "RND", "DIV", "DEN", "BND"],
        rows, "llcccc", r"\small"
    ))

    classical = pd.read_csv(EXP / "historical_primary_tables/descriptive_table.csv")
    rows = []
    for clf in CLFS:
        for method in CLASSIC_ORDER:
            match = classical[(classical.classifier == clf) & (classical.method == method)]
            if len(match) != 1:
                raise ValueError("Missing classical descriptive value")
            record = match.iloc[0]
            rows.append([CLF_LABEL[clf], CLASSIC_LABEL[method],
                         str(int(record.n_datasets)),
                         *[f3(record[m]) for m in METRICS]])
    write("table4.tex", table(
        "Complete-dataset descriptive means of the historical BND rule and classical methods",
        "tab:4", ["Classifier", "Method", "$n$", "F1", "BAcc", "G-Mean", "ROC-AUC", "MCC"],
        rows, "llrrrrrr", r"\scriptsize"
    ))

    paired = pd.read_csv(EXP / "historical_primary_tables/classical_paired_comparisons.csv")
    rows = []
    for clf in CLFS:
        for metric in METRICS:
            match = paired[(paired.classifier == clf) &
                           (paired.metric == metric) & (paired.comparator == "SMOTE")]
            if len(match) != 1 or match.iloc[0].n_datasets != 44:
                raise ValueError("Missing matched BND-SMOTE comparison")
            record = match.iloc[0]
            metric_label = {"BalancedAccuracy": "BAcc", "GMean": "G-Mean",
                            "ROC_AUC": "ROC-AUC"}.get(metric, metric)
            wtl = f"{int(record.wins)}/{int(record.ties)}/{int(record.losses)}"
            rows.append([CLF_LABEL[clf], metric_label,
                         signed(record.mean_difference), wtl,
                         p3(record.p_holm_all135)])
    write("table5.tex", table(
        "Paired historical BND minus standalone SMOTE across 44 complete KEEL datasets",
        "tab:5", ["Classifier", "Metric", "Mean difference", "W/T/L", "Holm $p$"],
        rows, "llrrr", r"\small"
    ))

    favorable = paired[(paired.comparator != "Original") &
                       (paired.p_holm_all135 < 0.05)].copy()
    if len(favorable) != 7 or (favorable.mean_difference <= 0).any():
        raise ValueError("Unexpected all-135 robust comparison set")
    rows = []
    for _, record in favorable.iterrows():
        metric_label = {"BalancedAccuracy": "BAcc", "GMean": "G-Mean",
                        "ROC_AUC": "ROC-AUC"}.get(record.metric, record.metric)
        rows.append([CLF_LABEL[record.classifier], metric_label,
                     CLASSIC_LABEL[record.comparator], str(int(record.n_datasets)),
                     signed(record.mean_difference), p3(record.p_holm_all135)])
    write("table6.tex", table(
        "Positive matched historical BND differences surviving Holm correction across all 135 classical contrasts",
        "tab:6", ["Classifier", "Metric", "Comparator", "$n$", "Difference", "Holm $p$"],
        rows, "lllrrr", r"\small"
    ))

    pool = pd.read_csv(EXP / "historical_pool_ablation_run/paired_pool_comparisons.csv")
    pool = pool[(pool.metric == "GMean") & (pool.comparator == "ONLY_SMOTE")]
    if len(pool) != 3:
        raise ValueError("Incomplete pool comparison")
    rows = []
    for clf in CLFS:
        record = pool[pool.classifier == clf].iloc[0]
        rows.append([CLF_LABEL[clf], f3(record.full_mean_on_pair),
                     f3(record.comparator_mean_on_pair),
                     signed(record.full_minus_comparator),
                     p3(record.p_holm_primary3)])
    write("table7.tex", table(
        "Historical BND on the full four-generator pool versus the SMOTE-only pool: G-Mean",
        "tab:7", ["Classifier", "Full pool", "SMOTE-only", "Difference", "Holm $p$"],
        rows, "lrrrr", r"\small"
    ))

    external = external_panel()
    rows = []
    for _, record in external.sort_values(["provenance", "dataset", "classifier"]).iterrows():
        rows.append([name_tex(record.dataset), CLF_LABEL[record.classifier],
                     f3(record.historical), f3(record.comparator_value),
                     signed(record.difference), record.provenance])
    write("table8.tex", table(
        "External five-fold G-Mean of historical BND versus standalone SMOTE",
        "tab:8", ["Source", "Classifier", "BND", "SMOTE", "Difference", "Protocol group"],
        rows, r"p{0.21\linewidth}lrrrl", r"\footnotesize"
    ))

    summary = {
        "table3_rows": 15, "table4_rows": 30, "table5_rows": 15,
        "table6_rows": len(favorable), "table7_rows": 3, "table8_rows": len(external),
        "external_BND_GMean_wins": int((external.difference > 1e-12).sum()),
        "external_BND_GMean_losses": int((external.difference < -1e-12).sum()),
        "external_mean_difference": float(external.difference.mean()),
    }
    (OUT / "table_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
