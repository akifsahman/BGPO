"""Check hashes, source snapshots, result dimensions, and upload archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def rows(path: Path) -> int:
    with path.open(newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent)
    root = parser.parse_args().root.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "MANIFEST.json" and
              p.name != ".DS_Store" and
              "__pycache__" not in p.parts and ".git" not in p.parts and
              p.suffix != ".pyc"}
    listed = {record["path"] for record in manifest["files"]}
    if actual != listed or len(listed) != manifest["file_count"]:
        raise AssertionError(f"Manifest paths differ: +{len(actual-listed)}, -{len(listed-actual)}")
    for record in manifest["files"]:
        data = (root / record["path"]).read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise AssertionError(f"Checksum changed: {record['path']}")
    keel = list((root / "Pool-Select/datasets_keel").rglob("*.dat"))
    if len(keel) != 440:
        raise AssertionError(f"Expected 440 KEEL fold files, found {len(keel)}")
    exp = root / "revision/experiments"
    snapshots = 0
    for folder in ("external_data", "adaptive_external_data"):
        for metadata_file in (exp / folder).glob("*.json"):
            metadata = json.loads(metadata_file.read_text())
            data = metadata_file.with_suffix(".csv")
            if hashlib.sha256(data.read_bytes()).hexdigest() != metadata["sha256_csv"]:
                raise AssertionError(f"OpenML snapshot changed: {data}")
            snapshots += 1
    if snapshots != 7:
        raise AssertionError(f"Expected seven OpenML snapshots, found {snapshots}")
    expected_rows = {
        "historical_family_run/fold_results.csv": 2640,
        "historical_probability_run/fold_results.csv": 660,
        "adaptive_external_run/fold_results.csv": 240,
    }
    for name, expected in expected_rows.items():
        count = rows(exp / name)
        if count != expected:
            raise AssertionError(f"{name}: {count}, expected {expected}")
    table = json.loads((root / "revision/historical_primary/tables/table_manifest.json").read_text())
    if table["table8_rows"] != 21 or table["external_BND_GMean_wins"] != 10:
        raise AssertionError("External table manifest changed")
    archive_path = root.parent / "github_upload.zip"
    if archive_path.exists():
        with ZipFile(archive_path) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != actual | {"MANIFEST.json"}:
                raise AssertionError("Upload ZIP differs from the GitHub folder")
    print(json.dumps({"verified_files": len(listed), "keel_files": len(keel),
                      "openml_snapshots": snapshots, "result_rows": expected_rows,
                      "upload_zip": str(archive_path)}, indent=2))

if __name__ == "__main__":
    main()
