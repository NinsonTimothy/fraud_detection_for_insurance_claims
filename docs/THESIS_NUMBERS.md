# Thesis number update sheet (auto-generated — do not hand-edit)

OLD = what `Project_documentation_updated.docx` currently says. NEW = read from the artifacts on disk when this file was generated (`python -m app.ml.generate_thesis_numbers`). Section numbers follow the Sept 2026 rewrite.

| Thesis location | What | OLD | NEW |
|---|---|---|---|
| 3.4.1 / Abstract | Engineered feature count | 70 | 69 |
| 3.3.6 | `is_no_witness` feature | included | removed (dataset artefact — see LIMITATIONS) |
| 3.5 | Champion selection data | paired 5-fold CV on all 1,000 rows | training split only (800 rows); the 200-row test split is never used for selection |
| 3.5 | Selection CV design | 5 folds, fixed 0.5 threshold | 5-fold stratified x 3 repeats = 15 paired folds; 5-fold, picks hyperparameters (PR-AUC) and threshold (F1) per candidate |
| 3.5 | Baseline | none | one-line rule: flag if incident_severity = Major Damage |
| 3.5 / 4.2 | Significance test | paired t-test, 4 df | Nadeau-Bengio corrected resampled t-test (two-sided) on the paired fold differences |
| 4.2 | RF vs LR recall p-value | 0.0086 (stale, hardcoded) | 0.836 (not significant) |
| 4.2 | RF vs LR F1 p-value | 0.0166 | 0.036 (significant) |
| 4.2 (new) | Champion vs rule, F1 difference / p | — | -0.004 / 0.212 |
| 4.2 (new) | Nested-CV F1: champion vs rule | — | 0.624 vs 0.629 |
| 4.2 (new) | Test decisions identical to rule | — | 100.0% |
| Table 4.1 | Champion | Random Forest | random_forest |
| Table 4.1 | Operating threshold | 0.44 | 0.45 |
| Table 4.1 | Champion F1 / Recall / Precision | 0.679 / 73.5% / 63.2% | 0.679 / 73.5% / 63.2% |
| Table 4.1 | Champion ROC-AUC / PR-AUC | 0.794 / 0.545 | 0.792 / 0.546 |
| Table 4.1 | logistic_regression: thr / F1 / Recall / ROC-AUC | see old table | 0.55 / 0.667 / 71.4% / 0.830 |
| Table 4.1 | xgboost: thr / F1 / Recall / ROC-AUC | see old table | 0.45 / 0.673 / 71.4% / 0.816 |
| Table 4.1 | major_damage_rule: thr / F1 / Recall / ROC-AUC | — (new row) | 0.05 / 0.679 / 73.5% / 0.798 |
| 4.4 | Oracle ROC-AUC (95% CI) | 0.463 (0.443–0.481) | 0.461 (0.441–0.480) |
| 4.4 | Oracle wording | random / indistinguishable from random | significantly inverted |
| 4.4 | SHAP weight constant on Oracle | 91.7% | 93.0% |
| 4.5 | SHAP share: is_major_damage + incident_severity_ordinal | 58.8% | 60.6% |
| 4.6 | High band: share of test claims | 26.5% | 27.5% |
| 4.6 | High band: share of fraud captured | 67.3% | 71.4% |
| 4.6 | Claims flagged at operating threshold / recall | 57 / 73.5% | 57 / 73.5% |
| 3.6 | Low band action wording | auto-approved | recommend standard claims handling — handler decides |
| 3.8 | Test count | 153 | backend: 166 tests collected in 2.53s; dashboard: 14 tests collected in 0.54s |

Text changes that go with the numbers:

- Ch. 4/5: state plainly that no ML model beat the Major-Damage rule on F1; the model's contribution is ranking within groups and explanation.
- Ch. 4.4 / 5: replace every 'random' description of Oracle with 'significantly inverted (95% CI entirely below 0.5)'.
- Ch. 5 Limitations: add the witness artefact paragraph from docs/LIMITATIONS.md.
- Ch. 3.6/3.7: decision-support wording — the system recommends, the investigator decides.
- Ch. 3.5: describe the earlier test-set leak in model selection and how it was fixed.
