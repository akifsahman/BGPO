"""Snapshot four metadata-frozen OpenML sources before adaptive modeling."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from reference_benchmark import ROOT


OUTPUT = ROOT / "revision/experiments/adaptive_external_data"
SOURCES = {
    "ozone-level-8hr": (1487, "1"),
    "phoneme": (1489, "1"),
    "musk": (1116, "1"),
    "seismic-bumps": (46956, "4"),
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, (data_id, expected_version) in SOURCES.items():
        csv = OUTPUT / f"{name}.csv"
        meta = OUTPUT / f"{name}.json"
        if csv.exists() and meta.exists():
            print(f"Existing snapshot: {name}", flush=True)
            continue
        bunch = fetch_openml(data_id=data_id, as_frame=True,
                             data_home=OUTPUT / "openml_cache")
        version = str(bunch.details.get("version"))
        if version != expected_version:
            raise ValueError(f"Unexpected OpenML version for {name}: {version}")
        X = bunch.data.copy()
        y = pd.Series(bunch.target, name="__target__").astype(str)
        if len(X) != len(y) or X.empty:
            raise ValueError(f"Invalid source table: {name}")
        excluded = []
        for column in X:
            values = pd.to_numeric(X[column], errors="coerce")
            if values.notna().sum() == 0:
                excluded.append(str(column))
            elif X[column].notna().sum() != values.notna().sum():
                raise ValueError(f"Mixed nonnumeric predictor: {name} {column}")
            else:
                X[column] = values
        X = X.drop(columns=excluded)
        if X.empty:
            raise ValueError(f"No numeric predictors in {name}")
        counts = y.value_counts(dropna=False)
        if len(counts) != 2 or counts.iloc[0] == counts.iloc[1]:
            raise ValueError(f"Nonbinary or balanced target: {name}")
        minority = sorted(counts.index, key=lambda label: (counts[label], label))[0]
        binary = (y == minority).astype(int)
        if binary.sum() < 10 or binary.sum() >= len(binary) / 2:
            raise ValueError(f"Invalid minority count: {name}")
        frame = X.reset_index(drop=True)
        frame["__target__"] = binary.to_numpy()
        frame.to_csv(csv, index=False)
        metadata = {
            "requested_name": name,
            "openml_data_id": data_id,
            "openml_source_name": bunch.details.get("name"),
            "openml_version": version,
            "source_url": f"https://www.openml.org/d/{data_id}",
            "sha256_csv": hashlib.sha256(csv.read_bytes()).hexdigest(),
            "observations": len(frame),
            "numeric_features": X.shape[1],
            "excluded_wholly_nonnumeric_columns": excluded,
            "original_target_counts": {str(k): int(v) for k, v in counts.items()},
            "minority_original_label": str(minority),
            "minority_count": int(binary.sum()),
            "majority_count": int((1 - binary).sum()),
            "missing_feature_cells": int(X.isna().sum().sum()),
            "feature_names": list(map(str, X.columns)),
        }
        meta.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved {name}: {len(frame)} rows, {X.shape[1]} numeric features, "
              f"minority {int(binary.sum())}", flush=True)


if __name__ == "__main__":
    main()
