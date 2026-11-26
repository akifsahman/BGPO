# Corrected 44-dataset primary comparison

This run evaluates the manuscript's deterministic, distance-based boundary selector on the supplied five-fold KEEL partitions. The historical workbook used a different boundary score and stochastic sampling; its values are not combined with this run.

## Protocol

- Data: `Pool-Select/datasets_keel`, 44 datasets × five paired train/test folds.
- Methods: Original, ROS, SMOTE, ADASYN, Borderline-SMOTE1/2, MWMOTE, G-SMOTE, K-Means SMOTE, and four selectors (RND, DIV, DEN, BND) acting on the same four-generator pool.
- Classifiers: RBF SVM, KNN with five neighbors, and RF with 100 trees.
- Budget: each successful augmented method adds exactly the training-fold majority minus minority count. Seed 42.
- Features: categorical mapping, median imputation, and standardization fit on training fold only; the paired test fold is transformed with those fitted values.
- Metrics: F1, balanced accuracy, G-Mean, ROC-AUC, and MCC. Fold means are then averaged within datasets.
- A method/dataset mean is considered complete only if all five folds were successful. No missing generator output is replaced or scored as zero.

Exact versions and classifier templates are in `protocol.json`. Raw input hashes are in `../../provenance.sha256`.

## Reproduce

From the project root with its `.venv` activated:

```bash
MPLCONFIGDIR=/private/tmp/bgpo-matplotlib .venv/bin/python revision/experiments/reference_benchmark.py
.venv/bin/python revision/experiments/analyze_full_run.py
.venv/bin/python revision/experiments/write_results.py
```

The benchmark saves each fold independently and resumes without repeating finished fold files. `--force` is required to recompute all fold outputs.

## Outputs and integrity

- `fold_results.csv`: 8,580 unique method/classifier/fold rows; 8,277 successful with finite metrics.
- `folds/`, `diagnostics/`: per-fold output and generator/selection diagnostics.
- `dataset_means_complete.csv`: 1,620 method/classifier/dataset rows with five successful folds.
- `descriptive_summary.csv`, `friedman_tests.csv`, `mean_ranks_common_sets.csv`, `paired_wilcoxon_holm.csv`: summary and paired analyses.
- `unavailable_methods.csv`: 3 ADASYN, 21 Borderline-SMOTE1, 21 Borderline-SMOTE2, and 56 K-Means SMOTE method/fold failures; all returned zero instead of the exact budget.

All four selectors completed all 44 datasets. The all-ten-method classical Friedman panel is complete on 24 datasets; each paired BND comparison uses its own complete matched dataset set. Holm adjustment is within each classifier/metric family, not across all 15 families. KEEL variants from shared source datasets are not independent, so statistical claims are exploratory.
