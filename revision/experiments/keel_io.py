"""Paired KEEL fold loading with train-fitted encoding and scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Fold:
    name: str
    number: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    nominal_columns: dict[str, tuple[str, ...]]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def _read_dat(path: Path) -> tuple[pd.DataFrame, list[str], str, set[str]]:
    attributes: list[str] = []
    nominal: set[str] = set()
    output: str | None = None
    rows: list[list[str]] = []
    in_data = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        lowered = line.lower()
        if lowered.startswith("@attribute"):
            match = re.match(r"@attribute\s+(\S+)\s+(.+)", line, re.I)
            if not match:
                raise ValueError(f"Malformed attribute in {path}: {line}")
            name, kind = match.groups()
            attributes.append(name)
            if "{" in kind and "}" in kind:
                nominal.add(name)
        elif lowered.startswith("@outputs"):
            output = line.split()[-1]
        elif lowered.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([part.strip() for part in line.split(",")])
    if not attributes or not rows:
        raise ValueError(f"No attributes or rows in {path}")
    if any(len(row) != len(attributes) for row in rows):
        raise ValueError(f"Row width differs from schema in {path}")
    frame = pd.DataFrame(rows, columns=attributes)
    output = output or attributes[-1]
    if output not in frame:
        raise ValueError(f"Output attribute absent in {path}")
    return frame, attributes, output, nominal


def load_fold(dataset_root: Path, name: str, number: int) -> Fold:
    folder = dataset_root / name
    train_path = folder / f"{name}-5-{number}tra.dat"
    test_path = folder / f"{name}-5-{number}tst.dat"
    train, train_columns, output, nominal = _read_dat(train_path)
    test, test_columns, test_output, test_nominal = _read_dat(test_path)
    if train_columns != test_columns or output != test_output or nominal != test_nominal:
        raise ValueError(f"Train/test schemas differ for {name} fold {number}")

    features = [column for column in train_columns if column != output]
    encoded_train = pd.DataFrame(index=train.index)
    encoded_test = pd.DataFrame(index=test.index)
    mappings: dict[str, tuple[str, ...]] = {}
    for column in features:
        if column in nominal:
            categories = tuple(sorted(train[column].unique()))
            mapping = {value: i for i, value in enumerate(categories)}
            unknown = set(test[column]) - set(mapping)
            if unknown:
                raise ValueError(f"Unseen categories in {name} fold {number}: {column}={unknown}")
            encoded_train[column] = train[column].map(mapping).astype(float)
            encoded_test[column] = test[column].map(mapping).astype(float)
            mappings[column] = categories
        else:
            a = pd.to_numeric(train[column].replace("?", np.nan), errors="raise")
            b = pd.to_numeric(test[column].replace("?", np.nan), errors="raise")
            median = a.median()
            if pd.isna(median):
                raise ValueError(f"All training values missing in {name}: {column}")
            encoded_train[column] = a.fillna(median).astype(float)
            encoded_test[column] = b.fillna(median).astype(float)

    labels, counts = np.unique(train[output], return_counts=True)
    if len(labels) != 2 or counts[0] == counts[1]:
        raise ValueError(f"Fold must have two unequally sized classes: {name} {number}")
    minority = labels[np.argmin(counts)]
    majority = labels[np.argmax(counts)]
    label_map = {majority: 0, minority: 1}
    if set(test[output]) - set(label_map):
        raise ValueError(f"Unseen test label in {name} fold {number}")
    y_train = train[output].map(label_map).to_numpy(dtype=int)
    y_test = test[output].map(label_map).to_numpy(dtype=int)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(encoded_train.to_numpy(dtype=float))
    X_test = scaler.transform(encoded_test.to_numpy(dtype=float))
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(f"Nonfinite features in {name} fold {number}")
    return Fold(
        name=name,
        number=number,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        nominal_columns=mappings,
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
    )
