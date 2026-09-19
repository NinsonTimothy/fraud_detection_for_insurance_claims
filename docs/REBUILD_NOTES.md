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
- Same architectural findings on `is_highrisk_hobby`/`is_exec_occupation` as
  dataset artifacts, and the external-validation collapse being a
  feature-availability problem, not an "Oracle is unlearnable" problem.
  (The `zip3_risk_tier` finding did NOT carry forward as "leakage, kept
  in" — see PB-02 below, it was removed entirely once the real bug was
  understood.)

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

**Historical record — pre-PB-02.** The table below was captured against the
model as it stood before the `zip3_risk_tier` 4-digit-prefix bug (§6 below)
was found and fixed. It is kept for the record of what the leak looked
like; it is not this build's current numbers. Current numbers live in
`models/metrics.json` and the README/CHANGELOG.

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

### 5. `incident_severity` ordinal encoding — confirmed, no longer a TODO (PB-24)

`ml_feature_critique.md` §"incident_severity" (written against the state of
this rebuild at the time) flags the ordinal order Trivial Damage < Minor
Damage < Major Damage < Total Loss as inferred, not confirmed, and
`feature_engineering.py` carried a matching `TODO-VERIFY` comment. That's
now resolved: the original FYP project's own `02_preprocessing.ipynb` /
`03_modelling.ipynb` encode `incident_severity` with `sklearn.OrdinalEncoder`
using this exact table, confirmed by reading that notebook's source
directly. The `TODO-VERIFY` comment has been removed from
`feature_engineering.py`; `ml_feature_critique.md` itself is left
unedited (it's carried forward verbatim as a point-in-time critique), so
this note is the correction of record.

`insured_education_level` stayed one-hot, deliberately NOT ordinal — the
original project's own ordinal order for it (JD < High School < ...)
ranks a doctoral law degree below a high-school diploma, which isn't a
real ordering worth reproducing.

### 6. `zip3_risk_tier` removed entirely — it was a 4-digit-prefix bug, not disclosed leakage (PB-02)

Sections 1-3 above (and the original `ml_feature_critique.md`) described
`zip3_risk_tier` as disclosed, self-documented target-encoding leakage —
"real, measurable, kept in because it's flagged" — the same framing as
`is_highrisk_hobby`/`is_exec_occupation`. That framing was itself wrong.

**Reproduction.** `feature_engineering.py` computed `zip3 = insured_zip //
100`. The comment assumed `insured_zip` was a genuine 5-digit US ZIP, so
`// 100` would produce a real 3-digit ZIP3 prefix (e.g. 10001 -> 100). But
this dataset's `insured_zip` values are 6-digit (e.g. 605280), so `// 100`
only drops the last two digits, leaving a 4-digit prefix (605280 -> 6052).
Reproduced directly against `data/cleaned/insurance_claims_cleaned.csv`:

- `insured_zip // 100` produces **515 distinct groups from 1,000 rows**
  (median group size 2.0; 90.9% of groups have <=3 rows) — essentially a
  per-row identifier, not a genuine geographic bucket.
- Building the target-encoded lookup from train and applying it to
  train/test separately showed near-total label memorization on train and
  a flat, uninformative rate on test: `zip3_risk_tier_low_risk` was 0.2%
  fraud on 55.4% of train rows vs. 28.2% on the equivalent test rows;
  `zip3_risk_tier_high_risk` was 71.8% fraud on 27.0% of train rows vs.
  20.0% on test.
- `data/processed/shap_feature_importance.csv` (pre-fix) showed the three
  `zip3_risk_tier_*` columns together accounted for **53.2% of total SHAP
  weight** (`high_risk` 25.8% + `low_risk` 25.7% + `medium_risk` 1.6%) —
  more than every other feature in the model combined.

This matches the ticket's own cited evidence, confirming the diagnosis.

**Fix.** Removed `zip3_risk_tier` (and the `_zip3_lookup_from_training()`
helper that built it) entirely from `feature_engineering.py`, `train.py`,
`inference.py`, `evaluate_oracle.py`, `backend/tests/test_ml_core.py`, and
the Model Insights dashboard page's text/SHAP-flag list. `insured_zip`
remains in `RAW_FEATURE_COLUMNS`/`MISSING_COLUMN_DEFAULTS` as an EDA-only
raw field — no feature is derived from it anymore. There is no honest
"recompute per-CV-fold" version of this feature worth keeping: even a
correctly-scoped 3-digit ZIP3 on a 1,000-row dataset would still be a
very high-cardinality, small-sample-per-bucket target encoding, and this
build's own high-cardinality-overfitting finding (§1 above, 122 -> 74
features) already argues against that pattern generally. Dropping it
outright is the same principle applied consistently, not a new one.

**Effect on metrics.** Retrained after this fix; see `models/metrics.json`
and the top of this repo's README/CHANGELOG for the resulting holdout and
CV numbers (generated from `metrics.json`, never hand-typed, per this
project's own working rules) — the §3 table above, written against the
PRE-PB-02 model, is left as historical record of what the leak looked
like, not as this build's current numbers.

### 7. `authorities_contacted` "None" was being read as missing data, not a real category (SH-01)

`clean_data.py`'s module docstring claimed `authorities_contacted` had 91
genuine `NaN` rows, mode-imputed to "Police". That claim was itself wrong.

**Reproduction.** `data/raw/insurance_claims_raw.csv` has 91 rows whose
`authorities_contacted` value is the literal text `None` — meaning "no
authority was contacted", the same kind of real category as
"Police"/"Fire"/"Other"/"Ambulance". Pandas' `read_csv()` treats the
string `"None"` as one of its own default NA-sentinel values, so a plain
`pd.read_csv(RAW_PATH)` silently turned those 91 genuine "None" claims
into `NaN`. `clean_data.py`'s generic "genuine NaN -> mode impute" step
then overwrote all 91 of them with `"Police"` — fabricating "police was
contacted" for claims that actually said the opposite, and (since fraud
claims may disproportionately skip involving authorities) potentially
erasing real signal. Confirmed directly: reading the raw CSV with
`keep_default_na=False, na_values=[""]` instead of pandas' defaults drops
the NaN count on this column from 91 to 0, and no other column in the raw
dataset is affected (checked column-by-column).

**Fix.** `clean_data.py` now reads the raw CSV with `keep_default_na=False,
na_values=[""]`, so only genuinely empty cells count as missing and
`"None"` survives as its own category. Because the cleaned CSV round-trips
that category back out as the literal text `"None"`, every other place
that reads a raw-schema CSV needed the same fix to avoid re-introducing
the exact same bug one step downstream: `train.py::load_and_split()`,
`backend/tests/test_ml_core.py`'s `sample_data` fixture, the batch-scoring
API endpoint (`api/scoring.py::score_batch`), and the dashboard's batch
upload page (`app_pages/batch_review.py`). `authorities_contacted` was
already in `feature_engineering.py`'s `CATEGORICAL_COLUMNS`, so no new
code was needed to turn the now-preserved `"None"` category into an
`authorities_contacted_None` one-hot column — it falls out of the existing
one-hot encoding automatically. With this fix, the raw dataset has zero
genuine `NaN` cells in any column (verified directly), so the generic
"genuine NaNs" mode-imputation loop in `clean_data.py` is currently dead
code for this dataset — left in as a safety net for a future re-upload
that might contain real missing cells.

**Effect on metrics.** Retrained after this fix (on top of PB-02):
`n_features` 71 -> 72 (the new `authorities_contacted_None` column);
holdout ROC-AUC for the shipped Random Forest moved from 0.843628
(post-PB-02, pre-SH-01) to 0.844979 — a small, expected move since this
only restores one genuine category on one categorical column. See
`models/metrics.json` for the exact current numbers rather than a
hand-typed figure here.
