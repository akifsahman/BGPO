"""Build historical-code supplement from recorded dataset means and paired tests."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"
CLFS = ("SVM_RBF", "KNN", "RandomForest")
METRICS = ("F1", "BalancedAccuracy", "GMean", "ROC_AUC", "MCC")
SELECTORS = ("HIST_RND", "HIST_DIV", "HIST_DEN", "HIST_BND")

def esc(x):
    return str(x).replace("_", r"\_")

def tab(caption, label, headers, rows):
    cols = "l" * (len(headers) - 2) + "r" * 2
    head = " & ".join(headers) + r" \\" 
    return "\n".join([r"\begingroup\small\setlength{\tabcolsep}{4pt}",
        rf"\begin{{longtable}}{{{cols}}}", rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule",
        head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\end{longtable}", r"\endgroup", r"\clearpage", ""])

def main():
    out = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[a4paper,margin=15mm]{geometry}
\usepackage{amsmath,amssymb,array,booktabs,longtable}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}{\Large\bfseries Supplementary Information}\par\medskip
{\large Boundary-Guided Sample Selection from a Composite Synthetic Pool for Imbalanced Classification}\par\medskip
Mehmet Akif \c{S}ahman\par Sel\c{c}uk University, Konya, Turkey\par
Correspondence: \texttt{asahman@selcuk.edu.tr}\end{center}
\noindent Dataset-level values are five-fold means. These algorithms reproduce the historical notebook with seed 42 and indexed pool positions.
"""]
    out.append((HERE / "supplementary_algorithms.tex").read_text())
    out.append(r"\section*{Dataset-level selector outcomes}")
    ds = pd.read_csv(EXP / "historical_family_run/dataset_means.csv")
    for idx, clf in enumerate(CLFS, 1):
        subset = ds[ds.classifier == clf]
        rows = []
        for dataset in sorted(subset.dataset.unique()):
            for sel in SELECTORS:
                r = subset[(subset.dataset == dataset) & (subset.method == sel)]
                if len(r) != 1:
                    raise ValueError((dataset, clf, sel))
                v = r.iloc[0]
                rows.append([esc(dataset), esc(sel.replace("HIST_", "")),
                    *[f"{v[m]:.3f}" for m in METRICS]])
        out.append(tab(f"Five-fold results for {esc(clf)} on 44 KEEL datasets",
            f"tab:s{idx}", ["Dataset", "Rule", "F1", "BAcc", "G-Mean", "ROC-AUC", "MCC"], rows))
    p = pd.read_csv(EXP / "historical_family_run/paired_comparisons.csv")
    rows = [[esc(r.classifier), esc(r.metric), esc(r.comparator),
        f"{r.mean_difference:+.4f}", f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}",
        f"{r.p_holm_all60:.4f}"] for _,r in p.iterrows()]
    out.append(tab("Historical BND minus other selectors and SMOTE: 60 adjusted comparisons",
        "tab:s4", ["Classifier", "Metric", "Comparator", "Difference", "W/T/L", "Holm $p$"], rows))
    p = pd.read_csv(EXP / "historical_primary_tables/classical_paired_comparisons.csv")
    rows = [[esc(r.classifier), esc(r.metric), esc(r.comparator), str(int(r.n_datasets)),
        f"{r.mean_difference:+.4f}", f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}",
        f"{r.p_holm_all135:.4f}"] for _,r in p.iterrows()]
    out.append(tab("Historical BND minus classical methods: 135 adjusted comparisons",
        "tab:s5", ["Classifier", "Metric", "Comparator", "$n$", "Difference", "W/T/L", "Holm $p$"], rows))
    p = pd.read_csv(HERE / "tables/external_BND_vs_SMOTE.csv")
    rows = [[esc(r.dataset), esc(r.classifier), r.provenance,
        f"{r.historical:.4f}", f"{r.comparator_value:.4f}",
        f"{r.difference:+.4f}"] for _,r in p.iterrows()]
    out.append(tab("External G-Mean: historical BND and standalone SMOTE",
        "tab:s6", ["Source", "Classifier", "Protocol", "BND", "SMOTE", "Difference"], rows))
    p = pd.read_csv(EXP / "adaptive_external_run/dataset_means.csv")
    rows = []
    for (dataset, clf), group in p.groupby(["dataset", "classifier"], sort=True):
        vals = group.set_index("method")["GMean"]
        rows.append([esc(dataset), esc(clf), f"{vals['ADAPTIVE']:.4f}",
            f"{vals['SMOTE']:.4f}", f"{vals['ADAPTIVE']-vals['SMOTE']:+.4f}"])
    out.append(tab("Pre-fixed adaptive choice on four new sources: outer-fold G-Mean",
        "tab:s7", ["Source", "Classifier", "Adaptive", "SMOTE", "Difference"], rows))
    out.append(r"\end{document}")
    (HERE / "supplementary.tex").write_text("\n\n".join(out))

if __name__ == "__main__":
    main()
