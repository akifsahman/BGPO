# Decision evidence: historical probabilistic BGPO versus manuscript BGPO

This is an exploratory method audit conducted after the original and corrected outcomes were already known. The historical selector is the algorithm actually called by the original notebook. It uses the class proportions among ten nearest pooled-class training neighbors to assign boundary weight and draws a fixed number of candidates without replacement using `RandomState(42)`. The submitted manuscript instead describes separate class-conditional mean distances and deterministic top-score selection.

## Paired 44-KEEL comparison

The historical selector was re-evaluated on the same five predefined folds, train-fitted preprocessing, four-generator pool, exact augmentation budget, three classifiers, and five metrics as the corrected benchmark. The historical BND rule was verified against its original function on a pilot fold, and the deterministic reference results were reproduced in all 220 folds. All 660 historical method/classifier/fold rows completed. Its selected indices overlapped deterministic BND by 33.8% on average.

| Classifier | Historical BND G-Mean | Deterministic BND | Historical DIV | SMOTE |
| --- | ---: | ---: | ---: | ---: |
| SVM (RBF) | 0.823 | 0.801 | 0.837 | 0.833 |
| KNN | 0.841 | 0.817 | 0.841 | 0.841 |
| Random Forest | 0.779 | 0.735 | 0.797 | 0.789 |

Historical BND improved balanced accuracy and G-Mean over deterministic BND for all three classifiers; those six differences survived Holm correction across 75 exploratory comparisons. It did not significantly outperform the historical RND or DIV selector or standalone SMOTE. When all four selectors from the original notebook were rerun on the corrected protocol, historical BND significantly beat historical DEN in KNN and RF balanced accuracy and G-Mean after correction across 60 tests, but not historical RND or DIV. SVM and RF historical DIV had higher average G-Mean than historical BND. Thus the old paper's favorable ranking does not transfer intact to the leakage-safe paired protocol.

Five historical-selection seeds (42--46) were examined after seeing the initial result. Historical BND exceeded deterministic BND on average G-Mean with all five seeds for all three classifiers. It remained below historical DIV for SVM and RF with all five seeds; KNN was near a tie. It stayed below SMOTE in SVM and RF with all five seeds. This is a robustness description, not independent confirmation.

## Three larger OpenML sources

The original external datasets and folds were reused **after** their earlier outcomes had been seen, so this is another exploratory check. All 45 method/classifier/fold rows completed, and recomputed deterministic BND metrics matched saved results on every fold. Historical BND exceeded deterministic BND and BND-Quota in all nine dataset/classifier G-Mean blocks. Against SMOTE it was higher in five blocks and lower in four; against DIV it was higher in three and lower in six. The mean difference across the nine blocks was +0.021 versus BND, +0.012 versus BND-Quota, +0.002 versus SMOTE, and -0.004 versus DIV. Three sources do not support a general superiority claim.

## Recommendation

The historical stochastic rule is a stronger **BND alternative** under the matched protocol and is the actual provenance of the originally submitted results. It does not show a reliable advantage over simple oversampling, uniform selection, or diversity selection. Switching the main paper solely to claim better performance would not be justified. If the authors choose to restore the historical rule as the stated method for provenance reasons, the entire method, equations, pseudocode, result tables, and reviewer response must be rewritten around the fresh matched runs; the original workbook must not be reused as if it followed the corrected protocol. The change must be disclosed as a correction of the submitted method description. Keeping the current distance-based method as primary, with the historical rule in a clearly labeled exploratory sensitivity analysis, is the narrower editorial choice.

Raw outputs: `historical_probability_run/`, `historical_seed_run/`, `historical_family_run/`, and `historical_external_run/`; each has its own protocol or audit records.
