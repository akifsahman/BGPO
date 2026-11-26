# Exploratory external check of the historical stochastic boundary rule

This check was specified after seeing the corrected KEEL and prior external outcomes. It is **exploratory**, not independent confirmation. The three source files, IDs, target constructions, five stratified fold indices, imputation/scaling, exact budget, pool generation, classifier templates, metrics, and seed 42 are unchanged from `external_validation_protocol.md`.

The one substituted method is the historical notebook's ten-nearest *combined-class* proportion score, followed by probability-weighted selection without replacement using `RandomState(42)`. Its selected rows are compared descriptively with the saved BND, DIV, SMOTE, and BND-Quota results. The primary descriptive metric is G-Mean; F1 and balanced accuracy are also inspected. No dataset, metric, method, or seed will be selected after viewing this run, and no significance claim is made from three sources.
