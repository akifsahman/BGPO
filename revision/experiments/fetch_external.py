"""Snapshot the three prespecified OpenML tables before external evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from reference_benchmark import ROOT


OUTPUT = ROOT / "revision/experiments/external_data"
DATASETS = {"mammography": 310, "MagicTelescope": 1120, "satimage": 182}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data_id in DATASETS.items():
        csv = OUTPUT / f"{name}.csv"
        meta = OUTPUT / f"{name}.json"
        if csv.exists() and meta.exists():
            print(f"Existing snapshot: {name}", flush=True)
            continue
        bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=OUTPUT / "openml_cache")
        X = bunch.data.copy()
        y = pd.Series(bunch.target, name="__target__").astype(str)
        if len(X) != len(y) or X.empty:
            raise ValueError(f"Invalid source table: {name}")
        excluded = []
        for col in X:
            values = pd.to_numeric(X[col], errors="coerce")
            if values.notna().sum() == 0:
                excluded.append(col)
            elif X[col].notna().sum() != values.notna().sum():
                raise ValueError(f"Mixed nonnumeric feature in {name}: {col}")
            else:
                X[col] = values
        if excluded:
            X = X.drop(columns=excluded)
        if X.empty:
            raise ValueError(f"No numeric predictors: {name}")
        labels = y.value_counts(dropna=False)
        if name == "satimage":
            rare = sorted(labels.index, key=lambda label: (labels[label], label))[0]
        elif len(labels) == 2:
            rare = sorted(labels.index, key=lambda label: (labels[label], label))[0]
        else:
            raise ValueError(f"Expected binary target in {name}: {labels.to_dict()}")
        binary = (y == rare).astype(int)
        if binary.sum() < 10 or binary.sum() >= len(binary) / 2:
            raise ValueError(f"Invalid minority count: {name}")
        frame = X.reset_index(drop=True)
        frame["__target__"] = binary.to_numpy()
        frame.to_csv(csv, index=False)
        sha = hashlib.sha256(csv.read_bytes()).hexdigest()
        metadata = {
            "requested_name": name,
            "openml_data_id": data_id,
            "openml_source_name": bunch.details.get("name"),
            "openml_version": bunch.details.get("version"),
            "source_url": f"https://www.openml.org/d/{data_id}",
            "sha256_csv": sha,
            "observations": len(frame),
            "numeric_features": len(X.columns),
            "excluded_all_nonnumeric_columns": excluded,
            "original_target_counts": {str(k): int(v) for k, v in labels.items()},
            "minority_original_label": str(rare),
            "minority_count": int(binary.sum()),
            "majority_count": int((1 - binary).sum()),
            "missing_feature_cells": int(X.isna().sum().sum()),
            "feature_names": list(map(str, X.columns)),
        }
        meta.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {name}: {len(frame)} rows, {len(X.columns)} features, minority {binary.sum()}", flush=True)


if __name__ == "__main__":
    main()
