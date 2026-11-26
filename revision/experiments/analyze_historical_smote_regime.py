"""Exploratory training-only overlap association with historical BND vs SMOTE."""

from __future__ import annotations

import json
import re

import pandas as pd
from scipy.stats import spearmanr

from analyze_full_run import holm_adjust
from reference_benchmark import ROOT


OUT = ROOT / "revision/experiments/historical_smote_regime"


def family(name: str) -> str:
    for prefix in ("new-thyroid", "page-blocks"):
        if name.startswith(prefix):
            return prefix
    match = re.match(r"[A-Za-z]+", name)
    if not match:
        raise ValueError(f"Unrecognized source name: {name}")
    return match.group().lower()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    overlap = pd.read_csv(
        ROOT / "revision/experiments/pool_overlap_run/dataset_overlap.csv"
    )[["dataset", "minority_majority_closer_share"]]
    historical = pd.read_csv(
        ROOT / "revision/experiments/historical_family_run/dataset_means.csv"
    )
    historical = historical[historical.method == "HIST_BND"]
    smote = pd.read_csv(
        ROOT / "revision/experiments/historical_primary_tables/complete_dataset_means.csv"
    )
    smote = smote[smote.method == "SMOTE"]
    panel = historical.merge(smote, on=["dataset", "classifier"],
                             suffixes=("_historical", "_smote"))
    panel = panel.merge(overlap, on="dataset")
    if len(panel) != 132 or panel.isna().any().any():
        raise ValueError("Incomplete historical-vs-SMOTE overlap panel")
    panel["GMean_delta"] = panel.GMean_historical - panel.GMean_smote
    panel["source_family"] = panel.dataset.map(family)
    panel.to_csv(OUT / "dataset_panel.csv", index=False)
    rows = []
    for classifier, subset in panel.groupby("classifier"):
        by_family = subset.groupby("source_family", as_index=False)[
            ["minority_majority_closer_share", "GMean_delta"]
        ].mean()
        if len(by_family) != 14:
            raise ValueError("Unexpected number of source families")
        rho, p = spearmanr(by_family.minority_majority_closer_share,
                           by_family.GMean_delta)
        rows.append({"classifier": classifier, "n_datasets": len(subset),
                     "n_source_families": len(by_family),
                     "family_spearman_rho": float(rho), "p_raw": float(p),
                     "mean_GMean_delta_all": float(subset.GMean_delta.mean()),
                     "mean_GMean_delta_overlap_le_0_5": float(subset.loc[
                         subset.minority_majority_closer_share <= 0.5,
                         "GMean_delta"].mean()),
                     "n_overlap_le_0_5": int((subset.minority_majority_closer_share <= 0.5).sum())})
    result = pd.DataFrame(rows)
    result["p_holm_three_family_tests"] = holm_adjust(result.p_raw.tolist())
    result.to_csv(OUT / "associations.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "status": "post-hoc exploratory, no threshold or gate validated",
        "associations": result.to_dict(orient="records"),
    }, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
