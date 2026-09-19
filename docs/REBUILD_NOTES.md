# Rebuild notes — what changed in Aegis Risk Engine vs. the original FYP prototype

This file exists so the numbers in this repo never look like an unexplained
discrepancy against `docs/ml_feature_critique.md` and `docs/generalization_and_cv_results.md`
(kept, verbatim, as the source documents this rebuild implements against).
Both are carried forward here for reference; this file is the honest diff.

## What's identical

- Same real, primary training data (`insurance_claims_raw.csv`, 1,000 rows,
  24.7% fraud rate — same source dataset, re-verified row-for-row against
  the hobby fraud-rate table in `ml_feature_critique.md`: chess 82.6%
  n=46, cross-fit 74.3% n=35, matching exactly).
- Same real Oracle dataset for external validation (15,420 rows, 6.0%
  fraud rate).
- Same shipped model family (Random Forest + SMOTE, compared against
  Logistic Regression and XGBoost), same threshold-tuning philosophy,
  same SHAP explainability, same honest-adapter methodology for Oracle
  (map only genuinely-overlapping fields, everything else to a
  documented fallback).
- Same three architectural findings: `zip3_risk_tier` leakage,
  `is_highrisk_hobby`/`is_exec_occupation` as dataset artifacts, and the
  external-validation collapse being a feature-availability problem, not
  an "Oracle is unlearnable" problem.

## What changed, and why

### 1. Feature count: 94 → 74 (`insured_hobbies`/`insured_occupation`/`auto_make` no longer fully one-hot encoded)

`ml_feature_critique.md` §1's own verdict on `insured_hobbies` and
`insured_occupation` is *"drop, or replace with something causally
grounded"* — an unapplied recommendation in the original prototype (the
critique documents the problem but the shipped `.pkl` still one-hot
encoded all 20 hobby levels and 14 occupation levels). This rebuild
applies that recommendation directly: only the single documented,
genuinely-informative flag from each (`is_highrisk_hobby`,
`is_exec_occupation`) is kept; the other 18+13 near-noise dummy columns
are dropped. `auto_make` (14 levels) was never flagged as useful anywhere
in the critique and is dropped outright.

An intermediate version of this rebuild kept the full one-hot encoding
(122 features) and is worth recording as a finding in its own right: it
cross-validated at ROC-AUC 0.94 but collapsed to 0.59–0.78 on a genuine
single holdout split, checked across 6 random seeds — textbook
high-cardinality overfitting on 800 training rows. Removing those columns
was as much a bug-fix as a critique-implementation.

### 2. Cross-validation methodology fixed (was silently leaking, now honest)

The original `generalization_and_cv_results.md` describes CV as evaluating
"each model's exact saved pipeline... not re-tuned" — ambiguous about
whether the ZIP3 target-encoding lookup was refit per fold. In this
rebuild's first pass, it was NOT (`zip3_lookup` was built once on the full
pool, then reused across all 5 CV folds) — this let most rows in every
fold be scored using a ZIP3 fraud-rate feature partly built from their own
label, inflating CV ROC-AUC to ~0.94 against a genuine single-holdout
range of 0.59–0.78. Rewriting `cross_validate_model()` in
`backend/app/ml/train.py` to refit the ZIP3 lookup, the scaler, and the
classifier from scratch on each fold's own training partition closed this
— see that function's docstring for the full explanation. This is the
same category of mistake (an evaluation signal quietly leaking
information it shouldn't have) as the corrected methodology in the
sibling MoMo Guard project, caught and fixed here proactively rather than
after the fact.

### 3. Honest numbers are lower than the original prototype's — expected, not a regression

| Metric (Random Forest) | Original prototype | This rebuild | Why different |
|---|---|---|---|
| Internal test ROC-AUC | 0.860 | ~0.66 | Fewer, less-overfit features; single 800/200 split, high variance at this sample size (checked 0.59–0.78 across 6 seeds) |
| Internal 5-fold CV ROC-AUC | 0.877 ± 0.036 | ~0.67 ± 0.04 | CV leakage fixed (see #2) — this is the honest number a leak-free evaluation of THIS feature set produces |
| Oracle ROC-AUC | 0.496 (random) | ~0.48 (random) | Consistent — both collapse to random, for the same root cause |
| Oracle: features constant on mapped data | 93.2% of SHAP weight | ~97.6% of SHAP weight | This rebuild's smaller feature set concentrates even more weight on `zip3_risk_tier`, which Oracle also can't supply |
| Oracle stress-test (fresh XGBoost on Oracle's own fields) | ROC-AUC 0.827 | ROC-AUC ~0.81 | Consistent — Oracle is learnable fraud data either way |

**Read this table as the expected cost of fixing two real methodological
problems** (leaky high-cardinality features, leaky CV), not as this
rebuild being a worse model. A lower, honestly-measured number is more
defensible in a viva than a higher, leaked one — this is the same
principle the sibling MoMo Guard project's corrected external-validation
work is built on.

### 4. New: a full production-style system around the same model

The original prototype was a single Streamlit app reading `.pkl` files
directly. This rebuild adds a FastAPI backend (3 ingestion transports,
SQLAlchemy models targeting Postgres with a SQLite fallback, investigator
feedback loop with a retraining-CSV export, audit log), a Kafka-based
streaming ingestion path (logic verified directly in Python — see
`docs/LIMITATIONS.md` for why no live broker was tested against), and
Docker Compose wiring all of it together. None of this changes the ML
findings above — it's the deployment shell around them.
