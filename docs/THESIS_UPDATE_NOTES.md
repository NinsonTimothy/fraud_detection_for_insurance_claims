# Thesis update notes (auto-generated — do not hand-edit)

Every number, table and claim in Chapters 3–5 that changes because of the pre-defence round-2 work. OLD = the Sept 2026 thesis. NEW = read from the named artifact by `python -m app.ml.generate_thesis_numbers`.

| Section | What | OLD | NEW | Source artifact |
|---|---|---|---|---|
| 3.4 | Engineered features | 70 | 69 | `models/metrics.json` |
| 3.3.6 | `is_no_witness` | included | removed (contradicts data; see LIMITATIONS) | `feature_engineering.py` |
| 3.4 | Imbalance handling | SMOTE + class weighting | class weighting only | `smote_vs_classweight_comparison.csv` |
| 3.5 | Selection data | CV over all 1,000 rows (test leaked) | development split only (800 rows); the 200-row test split is used once, for reporting | `champion_decision.json` |
| 3.5 | CV design | 5-fold, single run | repeated stratified 5-fold x 10 repeats = 50 paired folds | `champion_decision.json` |
| 3.5 | Thresholds in comparison | RF 0.44 vs LR/XGB 0.5 | each ML candidate: F1-optimal threshold on inner 5-fold out-of-fold predictions; rule: fixed (binary output) | `champion_decision.json` |
| 3.5 | Significance test | paired t-test (4 df) | Nadeau-Bengio corrected resampled t-test, two-sided, alpha=0.05 | `champion_decision.json` |
| 3.5 | Selection rule | Recall > F1 hierarchy; RF hardcoded | best mean PR-AUC among ML models, unless a simpler model (LR < RF < XGB) is not significantly worse on PR-AUC, in which case the simplest such model; the rule baseline is reported but not eligible | `champion_decision.json` |
| 3.5 | Baseline | none | rule: incident_severity == Major Damage | `model_selection.py` |
| 3.5/4.2 | Champion | Random Forest | logistic_regression | `models/metrics.json` |
| 3.5 | Calibration | none (scores called probabilities) | sigmoid | `calibration_summary.csv` |
| 3.6 | Risk bands | fixed 0.30 / 0.60 | 0.148 / 0.380 (derived) | `models/risk_policy.json` |
| 3.6 | Review threshold | 0.44 | 0.27 | `models/metrics.json` |
| 3.6 | Cost-optimal threshold | 0.07 (presented as optimal) | sensitivity grid: 0.04–0.80 over 36 assumption sets | `cost_sensitivity.csv` |
| 3.6 | Low-band action | No action — auto-approved | Low priority — standard processing; an investigator may still review | `risk_policy.py` |
| 4.2 | RF vs LR recall p | 0.0086 (stale, hardcoded) | 0.839 | `pairwise_tests.csv` |
| 4.2 | RF vs LR F1 p | 0.0166 | 0.006 | `pairwise_tests.csv` |
| 4.2 (new) | Champion minus rule, pr_auc | — | +0.038, p = 0.076 | `champion_decision.json` |
| 4.2 (new) | Champion minus rule, recall | — | -0.007, p = 0.832 | `champion_decision.json` |
| 4.2 (new) | Champion minus rule, f1 | — | -0.056, p = 0.005 | `champion_decision.json` |
| Table 4.1 | Champion threshold / recall / precision / F1 | RF 0.44 / 73.5% / 63.2% / 0.679 | logistic_regression 0.27 / 69.4% / 55.7% / 0.618 | `model_comparison.csv` |
| Table 4.1 | Champion PR-AUC / ROC-AUC | 0.545 / 0.794 | 0.558 / 0.816 | `model_comparison.csv` |
| Table 4.1 | random_forest: thr / F1 / PR-AUC / ROC-AUC | see thesis | 0.45 / 0.673 / 0.518 / 0.767 | `model_comparison.csv` |
| Table 4.1 | xgboost: thr / F1 / PR-AUC / ROC-AUC | see thesis | 0.39 / 0.596 / 0.482 / 0.779 | `model_comparison.csv` |
| Table 4.1 | major_damage_rule: thr / F1 / PR-AUC / ROC-AUC | — | 0.50 / 0.679 / 0.529 / 0.798 | `model_comparison.csv` |
| 4.3 (new) | Champion test Brier / ECE | not reported | 0.139 / 0.076 | `model_comparison.csv` |
| 4.2.4 | Proxy ablation PR-AUC off -> on | holdout comparison (used test set) | 0.527 -> 0.692 (dev CV) | `proxy_feature_ablation.csv` |
| 4.4 | Oracle ROC-AUC (95% CI) | 0.463 (0.443–0.481) | 0.519 (0.499–0.538) | `oracle_validation_report.json` |
| 4.4 | Oracle wording | random | no measurable ranking signal | `reporting.roc_ci_verdict` |
| 4.4 | Oracle fields mapped | 7 or 10 (inconsistent) | {'unmappable': 25, 'approximate': 5, 'direct': 4} | `oracle_field_mapping.csv` |
| 4.4 | SHAP weight constant on Oracle | 91.7% | 88.0% | `oracle_validation_report.json` |
| 4.5 | SHAP share of incident severity | 58.8% (two columns) | 19.7% | `shap_importance_by_field.csv` |
| 4.6 | High band: share of test claims / of fraud | 26.5% / 67.3% | 22.5% / 61.2% | `risk_scores_test.csv + risk_policy.json` |
| 4.6 | Claims flagged at review threshold / recall | 57 / 73.5% | 61 / 69.4% | `risk_scores_test.csv` |
| 4.6 | Test decisions identical to rule | 100% (not reported) | 96.0% | `models/metrics.json` |

## Claims to rewrite (not just numbers)

- **Champion.** The champion is `logistic_regression`, selected by a rule written in code before results (best PR-AUC unless a simpler model is not significantly worse). Remove wording that presents Random Forest as the measured winner.
- **Rule baseline.** State plainly what `champion_decision.json` says under `champion_significantly_beats_rule_on` and `rule_significantly_beats_champion_on`.
- **Scores** are calibrated fraud-risk scores reported with Brier and ECE; say 'fraud-risk score' throughout.
- **Oracle.** Replace 'random' with the CI-derived verdict ('no measurable ranking signal'); add the field-mapping and univariate-AUC tables.
- **Witnesses.** Add the fraud-rate-by-witnesses table to Limitations as a dataset artefact.
- **Cost model.** Present the cost threshold as a sensitivity analysis under stated assumptions, not an optimum.
- **Leakage.** Describe the test-set leak in model selection, proxy ablation and threshold choice, and its fix.
- **Decision support.** The system recommends a priority; investigators decide. No auto-approval anywhere.
- **Calibration rule amendment.** Disclose that the isotonic/sigmoid preference was amended after the first run.
