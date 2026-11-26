# BGPO: reproducible experiments for the revised manuscript

This repository contains the code, fixed data snapshots, fold-level results, and LaTeX sources for the revised BGPO study. The principal experiment compares candidate-selection rules with SMOTE and other baselines on 44 KEEL tasks. External-source analyses are reported separately.

The revised manuscript documents the **historical notebook boundary selector**: a ten-neighbor class-composition score followed by seeded weighted sampling without replacement. This differs from the distance-ranking rule described in the original submission. The revised tables and claims are tied to the included code and fold outputs; the two definitions should not be conflated.

## Repository layout

| Path | Contents |
| --- | --- |
| `revision/experiments/` | Benchmark and analysis scripts, protocols, and archived fold-level outputs |
| `revision/historical_primary/` | Revised manuscript, supplementary pseudocode, bibliography, and generated tables |
| `historical/notebook/` | Unmodified author-supplied notebook and its supporting Python script, retained for provenance |
| `Pool-Select/datasets_keel/` | 44 supplied KEEL datasets, each with five predefined train/test folds |
| `requirements.txt`, `requirements.lock` | Direct dependencies and the fuller recorded environment |
| `MANIFEST.json`, `verify_package.py` | File hashes and package-integrity checks |

The `historical/notebook/` directory has a publication-neutral name; its two files are byte-identical to the author-supplied originals. See `revision/experiments/HISTORICAL_CODE_PROVENANCE.md` for the methodological provenance and the distinction from the originally stated rule.

## Install and verify

Run these commands from the repository root with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_package.py
```

`verify_package.py` checks the manifest hashes, the 440 KEEL fold files, seven OpenML snapshots, selected fold-result row counts, and the external table manifest. It does not rerun the computationally expensive benchmarks.

## Reproduce tables from archived fold outputs

The fold outputs are already included. To regenerate the analyses and manuscript tables, run:

```bash
python revision/experiments/analyze_historical_family.py
python revision/experiments/analyze_historical_primary.py
python revision/experiments/analyze_historical_pool_ablation.py
python revision/experiments/analyze_historical_external.py
python revision/experiments/analyze_adaptive_external.py
python revision/historical_primary/generate_tables.py
python revision/historical_primary/build_supplement.py
```

These commands write into the included result and table directories. The source manuscript is `revision/historical_primary/main.tex`; supplementary methods and pseudocode are in `supplementary.tex` and `supplementary_algorithms.tex` in the same directory.

## Rerun experiments

The benchmark scripts resume from recorded folds. To obtain a completely fresh run, use a clean copy of this repository without the corresponding `*_run/` outputs, then run the benchmarks in this order:

```bash
python revision/experiments/reference_benchmark.py
python revision/experiments/historical_probability_benchmark.py
python revision/experiments/historical_family_benchmark.py
python revision/experiments/historical_pool_ablation_benchmark.py
python revision/experiments/external_validation.py
python revision/experiments/historical_external.py
python revision/experiments/adaptive_external_benchmark.py
python revision/experiments/historical_cost_paired.py
```

The full experiment is computationally expensive. Protocol files under `revision/experiments/` identify which analyses were exploratory and which external validation rule was fixed before outcomes were examined.

## Data and interpretation

The KEEL folds are under `Pool-Select/datasets_keel/` to keep script paths consistent with the archived experiment. The seven OpenML snapshots and metadata are in `revision/experiments/external_data/` and `revision/experiments/adaptive_external_data/`; metadata includes source IDs, versions, preprocessing decisions, and SHA-256 checksums. Original sources: [KEEL](https://sci2s.ugr.es/keel/datasets.php) and [OpenML](https://www.openml.org/).

The historical BND selector did not show overall superiority to SMOTE on the 44 KEEL tasks. The external panels and adaptive analysis are reported with their own scope and limitations in the manuscript. Related KEEL variants may share underlying data, which matters when interpreting aggregate tests.

## License and citation

`LICENSE.md` grants MIT terms to the original project code identified there. The KEEL and OpenML data, historical notebook, manuscript text, and third-party LaTeX files are outside that grant. Cite the manuscript and the original dataset and oversampling sources when using this work.
