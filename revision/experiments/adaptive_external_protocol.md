# Frozen nested-selection external protocol (25 November 2026)

The KEEL outcomes and the earlier mammography/MagicTelescope/satimage outcomes were inspected before this protocol. They are development and exploratory evidence. The four OpenML sources below were chosen from their public metadata **before any adaptive or comparator outcomes were computed**. They are distinct from those three and from the KEEL source families. No source will be replaced or excluded based on a model score.

| OpenML ID | Name | Expected version | Planned task |
| --- | --- | ---: | --- |
| 1487 | ozone-level-8hr | 1 | Native binary minority-versus-majority |
| 1489 | phoneme | 1 | Native binary minority-versus-majority |
| 1116 | musk | 1 | Native binary minority-versus-majority |
| 46956 | seismic-bumps | 4 | Native binary minority-versus-majority |

For each source, use the native binary target, assign its less frequent class positive (ties are invalid), retain numeric predictors, discard wholly nonnumeric predictor columns, and fail if a mixed numeric/nonnumeric predictor or fewer than 10 minority observations is encountered. Record source version, class counts, excluded columns, file hash, and any source failure before modeling. Missing numeric values receive a median imputed on the relevant training portion only.

Use five stratified **outer** folds with `shuffle=True, random_state=20261125`. For each outer fold, use three stratified **inner** folds on the outer-training rows with `shuffle=True, random_state=20261126`. At each inner split, fit median imputation and `StandardScaler` on inner training only; generate and select synthetic candidates on inner training only. Compare three fixed options: standalone SMOTE, the exact historical four-generator stochastic BND selector, and the exact historical four-generator DIV selector. Each must add exactly `majority-minus-minority` synthetic observations; failed options receive no score for that split and can be chosen only if all three inner splits succeed. Evaluate each option with the same classifier template, score inner G-Mean, and select the option with the highest mean across the three inner folds. Exact ties prefer SMOTE, then historical BND, then historical DIV. No threshold or hyperparameter is tuned from the outer tests.

Refit the selected method on the entire outer training portion with freshly fitted preprocessing and evaluate on the untouched outer test portion. Also score all three fixed methods on that same outer split for transparent diagnostics. Classifier templates remain RBF SVM (`C=1`, default gamma), 5-neighbor KNN, and 100-tree RF, with the existing seeds. Primary outcome is the adaptive-minus-standalone-SMOTE G-Mean difference summarized as one five-fold mean per external source and classifier. F1, balanced accuracy, ROC-AUC, MCC, selection counts, source failures, and total decision cost are secondary. With only four independent sources, report effect sizes and blocks descriptively; do not infer general superiority from a nominal p value.

The method is an adaptive **wrapper**, distinct from claiming that historical BND itself beats SMOTE. It incurs three inner evaluations of three candidates and must be compared with simpler validation-driven choices before a final practical-value claim. The earlier three external sources can be described separately but are not confirmatory for this adaptive rule.
