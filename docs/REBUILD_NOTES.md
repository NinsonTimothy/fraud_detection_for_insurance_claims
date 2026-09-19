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
like; it is not this build's current numbers, and the model has been
retrained several more times since this table was captured (PB-14's
champion comparison, PB-24's `incident_education_level` encoding fix,
among others), so even the "This rebuild" column below is itself now a
historical snapshot, not live. **Current numbers, always regenerated from
`models/metrics.json` and the Oracle validation report (never hand-typed):
run `python -m app.ml.generate_metrics_report` (PB-15) or read
`docs/CURRENT_METRICS.md`, its output.** (A concrete instance of exactly
the drift this rule exists to prevent was found while building PB-15: this
table's own "Oracle ROC-AUC ~0.48 (random)" row, below, no longer matches
the current model — see `docs/CURRENT_METRICS.md`'s bootstrap CI, which no
longer contains 0.5 at all.)

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

### 8. Threshold/SHAP/cost-sweep leakage from the test set, fixed (PB-03)

`train.py` used to compute three different things directly from
`X_test_scaled`/`y_test` — the same rows later used to report final
performance: (1) the F1-optimal `operating_threshold`, via a grid search
scored on the test set; (2) the cost-optimal threshold, via
`find_cost_optimal_threshold()` scored on the test set with test-set claim
amounts; (3) global SHAP importance, computed on the test set. Choosing a
threshold BY maximizing F1 on the test set and then reporting that
threshold's F1 ON the same test set is optimistic by construction — it
isn't an honest estimate of how the model performs at a threshold chosen
without having seen those rows.

**Fix.** Added `_out_of_fold_proba()` to `train.py`: a 5-fold
`StratifiedKFold` loop over the TRAINING set only, refitting a fresh
scaler + RF estimator per fold (same refit-per-fold pattern already used
by `cross_validate_model()`) and collecting each fold's held-out
predictions, so every training row gets an honest probability that its
own fold's model never trained on. `operating_threshold` and the
cost-optimal threshold are now both chosen from these out-of-fold
training probabilities; global SHAP importance is now computed on
`X_train_scaled` (the data the shipped model was actually fit on,
appropriate for describing what it learned) instead of the test set. The
true test set (`X_test_scaled`/`y_test`) is now touched exactly once, in
`evaluate()`, purely to report performance at a threshold chosen without
it — model_comparison.csv, the per-model recall/precision/F1/ROC-AUC/PR-AUC
table, is the only place test-set data flows into a number this project
reports.

**The "degenerate cost-optimal threshold" finding, explained, not
papered over.** With this fix, the cost-optimal threshold search lands at
**0.01 — the very bottom of the swept range** (recall 1.0, precision
0.2475), i.e. "flag literally every claim." This is not a leftover bug in
the sweep: this dataset's mean `total_claim_amount` (~$52,762) is ~211x
the flat false-positive review cost (`ANALYST_REVIEW_COST`, default
$250), so under a pure expected-cost objective, missing one extra real
fraud case is almost always worse than reviewing ~211 extra false alarms
— a naive cost-minimizing search will rationally push toward
near-universal flagging under that ratio, regardless of how honestly the
threshold is chosen. It is reported in `models/metrics.json` as a
diagnostic/sensitivity-analysis number (see `cost_threshold.py`'s module
docstring) precisely because it isn't a usable operating point on its
own — an investigator team cannot review "nearly every claim." The number
actually used to flag claims (`operating_threshold`) continues to be the
F1-optimal one.

**Effect on metrics.** Retrained after this fix (on top of PB-02/SH-01).
`operating_threshold` came out identical (0.30) to the pre-fix, leaky
version — the F1-optimal point turned out to be robust to whether it was
chosen from the test set or from honest out-of-fold training data, so the
model_comparison numbers are unchanged from §7's. What changed is the
METHODOLOGY, not (in this instance) the resulting number — but the
process is now honest regardless of whether a future retrain or
hyperparameter change happens to move that agreement.

### 9. `is_new_customer` was thresholding a near-degenerate field (SH-06)

`is_new_customer` used to be `policy_age_at_incident_days < 30`, where
`policy_age_at_incident_days = incident_date - policy_bind_date`. Reproduced
against the cleaned dataset: this field's mean is ~4,739 days (~13 years),
and its "new" group is near-empty at every day-threshold tested — 0.2% of
rows at 30 days, only 1.7% even at 180 days — with fraud rate actually
LOWER in the "new" group than the "old" group at every threshold under
90 days. `incident_date`/`policy_bind_date` in this dataset simply don't
encode a real short-tenure relationship, so no choice of day-threshold on
that field could have produced a usable feature — the ticket's framing
("threshold choice") undersold the actual problem: the wrong field was
being thresholded.

`months_as_customer` — a real, directly-supplied raw field (not derived
from two dates) — shows the documented "new policy, big claim" pattern
properly. A Fisher-exact-test grid over 6-36 month cutoffs:

| threshold | n (new) | fraud (new) | fraud (rest) | delta | Fisher p |
|---|---|---|---|---|---|
| 6mo | 13 (1.3%) | 15.4% | 24.8% | -0.094 | 0.746 |
| 12mo | 24 (2.4%) | 29.2% | 24.6% | +0.046 | 0.633 |
| 18mo | 32 (3.2%) | 34.4% | 24.4% | +0.100 | 0.212 |
| 21mo | 37 (3.7%) | 35.1% | 24.3% | +0.108 | 0.172 |
| 24mo | 41 (4.1%) | 34.1% | 24.3% | +0.099 | 0.194 |
| 36mo | 66 (6.6%) | 27.3% | 24.5% | +0.028 | 0.658 |

None reach conventional significance at this sample size (lowest p~=0.17
at 21 months) — reported honestly as a directionally-consistent but
low-confidence signal, not a strong one. **Fix:** `is_new_customer` is now
`months_as_customer < 24` (24 months, a standard "new business"
underwriting window, near the empirical peak and with a reasonably sized,
stable group). Also clipped `policy_age_at_incident_days` at 0 — one row
had a negative "age" (incident_date before policy_bind_date), a data
quality artifact, not a real pre-bind claim.

**Effect on metrics.** Retrained on top of PB-02/SH-01/PB-03: holdout
ROC-AUC for the shipped Random Forest moved to 0.849 (see
`models/metrics.json`), `n_features` unchanged at 72 (one column's
DEFINITION changed, not the count).

### 10. SMOTE + class_weight together: kept, evidence-backed (SH-03)

Every shipped/compared model uses BOTH SMOTE oversampling AND
`class_weight="balanced"`/`"balanced_subsample"` simultaneously — two
different techniques for the same imbalance problem, worth checking
whether one alone would do. `backend/app/ml/model_selection_experiments.py`
runs a paired (identical 5 folds), refit-per-fold CV comparing, for both
Random Forest and Logistic Regression: (a) SMOTE + class_weight
(current), (b) SMOTE only, (c) class_weight only, no SMOTE. Full numbers
in `data/processed/smote_vs_classweight_comparison.csv`; headline finding:

- **Logistic Regression:** (a) and (b) are numerically IDENTICAL
  (ROC-AUC 0.864975 both ways) — once SMOTE has balanced the classes to
  ~1:1, `class_weight="balanced"` computes near-uniform weights on the
  already-balanced resampled data, so it has no additional effect. Not a
  bug — a correct, if slightly redundant, interaction between the two
  techniques.
- **Random Forest:** the three variants differ only marginally (ROC-AUC
  0.8473 / 0.8469 / 0.8426 for both+SMOTE-only+classweight-only
  respectively; recall is IDENTICAL — 0.886776 — across all three at the
  0.5 threshold used for this comparison), because `balanced_subsample`
  still varies its per-tree bootstrap weighting even on SMOTE-resampled
  data, unlike LR's single global fit.

**Decision: keep the current combined SMOTE + class_weight approach.**
No variant meaningfully outperforms it, switching would add a decision
point without measurable benefit, and it matches the pattern already
validated across every other retrain in this rebuild. Documented here so
the combination is a checked, evidence-backed choice rather than an
untested default carried over without scrutiny.

### 11. Champion model: measured evidence overrides the D4 default (PB-14)

D4's default was "champion = regularised Logistic Regression ... unless
my own nested-CV disagrees, RF as challenger." `run_champion_comparison()`
in `model_selection_experiments.py` runs a paired (identical 5 folds
across every candidate), refit-per-fold CV for Random Forest, Logistic
Regression, and XGBoost, plus a paired t-test on recall and F1 — this
project's own top two priority metrics (Recall > F1 > PR-AUC > ROC-AUC >
Accuracy, see `README.md`'s "Success metrics, in priority order"). Full numbers in
`data/processed/champion_comparison_paired_cv.csv`; decision record in
`data/processed/champion_decision.json`.

**Result: it disagrees.** Random Forest wins both top-priority metrics:
recall 0.8868 vs. Logistic Regression's 0.8301 (paired t-test t=4.80,
**p=0.0086**), F1 0.7517 vs. 0.7223 (t=3.96, **p=0.0166**) — both below
the conventional 0.05 threshold, meaning Random Forest's advantage is
consistent in direction and magnitude across every one of the 5 folds,
not a fluke of one split (though a 5-fold paired t-test has only 4
degrees of freedom, so this is real, reproducible evidence for THIS
dataset, not a claim of certainty beyond it). Logistic Regression wins on
ROC-AUC (0.865 vs. 0.847) and PR-AUC (0.658 vs. 0.566) — both ranked
BELOW recall/F1 in this project's own stated hierarchy.

**Decision: Random Forest remains champion** (unchanged from every prior
retrain in this rebuild — `metrics.json`'s `primary_model` was already
`"random_forest"`, so no train.py/inference.py change was needed), with
Logistic Regression reported as the runner-up/challenger. This is exactly
the scenario D4 anticipated: the default (LR) is overridden because this
project's own measured, paired nested-CV evidence — read through this
project's own stated metric priorities, not a generic "highest ROC-AUC
wins" rule — disagrees with it.

### 12. Risk policy (score -> band -> flag -> action) unified into one module (PB-04)

`inference.py`'s `score_batch()` and `score_one()` each independently
computed the Low/Medium/High risk band from the fraud probability —
`score_batch()` via `pd.cut(proba, bins=[-0.01, 0.3, 0.6, 1.01])`,
`score_one()` via a hand-written `"High" if p >= 0.6 else "Medium" if p
>= 0.3 else "Low"`. Reproduced directly: `pd.cut`'s default bins are
right-inclusive (`(a, b]`), so `proba == 0.30` lands in the FIRST bin
("Low"); the hand-written version's `>=` puts `0.30` in "Medium" instead.
Same disagreement at `proba == 0.60` ("Medium" vs. "High"). The SAME
claim, scored through batch upload vs. the single-claim form/API, could
get a different risk_grade purely from which code path it went through.

**Fix.** Added `backend/app/ml/risk_policy.py`: one module owning the
band edges (`MEDIUM_RISK_EDGE = 0.3`, `HIGH_RISK_EDGE = 0.6`), a scalar
`grade_for()`, an array `grade_for_array()` that uses the exact same
`>=` comparisons (not `pd.cut`) so it cannot disagree with the scalar
version, `is_flagged()` (the separate flag/no-flag decision, driven by
`operating_threshold`, not the band edges), and `recommended_action()` (a
plain-language next step per band: Low -> auto-approved, Medium ->
standard review queue, High -> priority SIU escalation — new, not
persisted to the database, computed fresh wherever it's displayed).
`inference.py`'s `score_batch()`/`score_one()` both now call into this
one module; `recommended_action` was added to the `/score` API response
(`ScoreOut` schema) and to the Score-a-claim dashboard page.

Regression test: `backend/tests/test_risk_policy.py` — parametrized over
values straddling both band edges (`nextafter` on each side), confirms
`grade_for()` and `grade_for_array()` now always agree, confirms the
documented boundary semantics, and confirms `is_flagged()` stays decoupled
from the band edges (an operating threshold below `MEDIUM_RISK_EDGE` can
flag some "Low"-banded claims — that's intentional, not a bug). Manually
verified against 20 real claims from the cleaned dataset:
`score_batch()`/`score_one()` risk_grade now agrees on every row (0
mismatches).

Tests: 31/31 backend (17 existing + 14 new), 3/3 dashboard.

### 13. `/score` had no real input validation (PB-06)

`ClaimIn.payload` was `dict[str, Any]` — no type, range, or enum
checking, and no rejection of unknown keys. Reproduced directly against a
live `TestClient`, all four accepted with a silent 200:

  - A typo'd key (`"aage": 35`) was silently ignored; the real `age`
    silently fell back to its documented default (39) with no signal to
    the caller that their field name was wrong.
  - Garbage/out-of-range values (`age=-999`, `total_claim_amount=-50000`)
    were silently accepted and scored as if real.
  - A completely empty `{"payload": {}}` was silently accepted and
    scored — a fully-default, meaningless claim, status 200.
  - Worst: `{"age": "thirty-five"}` (wrong TYPE) was silently accepted.
    `feature_engineering.py`'s numeric-passthrough columns run through
    `pd.to_numeric(..., errors="coerce").fillna(0.0)`, so the unparseable
    string silently became `age=0` — not even the documented default, a
    different and worse silent failure than a normal missing field.

**Fix.** `backend/app/api/schemas.py`'s new `ClaimPayload` model gives
every one of `feature_engineering.RAW_FEATURE_COLUMNS` an explicit type,
a range (`Field(ge=..., le=...)`), or an enum (`Literal[...]`, built from
the actual distinct values in `data/cleaned/insurance_claims_cleaned.csv`
for the genuinely closed-vocabulary columns — `policy_state`,
`insured_sex`, `incident_severity`, etc.); `model_config =
ConfigDict(extra="forbid")` rejects any key that isn't a real raw field
(catching typos as a 422 instead of a silent default); a model validator
rejects a payload with no fields set at all. Every field stays Optional —
this project's own design (live API, batch CSV upload,
`oracle_adapter.py`'s external adapter) depends on supplying any SUBSET
of raw fields and letting `apply_missing_defaults()` fill the rest, so
"required" here means "correct if present", not "every field must be
supplied". Free-text fields the model only reduces to a single flag
(`insured_hobbies`, `insured_occupation`) and the unused `auto_make` stay
free strings rather than enums, since an unrecognized value there
degrades gracefully instead of silently corrupting a numeric feature.

Also added **defaulted-fields reporting**: `inference.py`'s `score_one()`
now returns `defaulted_fields` — the raw fields this specific claim did
NOT supply and therefore fell back to `MISSING_COLUMN_DEFAULTS` for — so
a caller can tell a real-data-backed score from a mostly-defaulted one
instead of the two looking identical in the response. Exposed via the
`/score` response (`ScoreOut.defaulted_fields`).

`backend/tests/test_input_validation.py` (new) locks in all four
reproduced bugs as 422s, plus enum/date-format rejection and the
defaulted-fields report on a genuinely partial claim. Scope: this ticket
covers the single-claim `/score` JSON endpoint only — batch CSV upload
(`/score/batch`) and the Kafka ingestion path are unchanged (the latter
is slated for full removal under D1/PB-08/PB-09; validating a path about
to be deleted would be wasted work).

Tests: 39/39 backend (31 existing + 8 new), 3/3 dashboard. No retrain
needed — this ticket only touches the API request-validation layer.

### 14. Expected client mistakes were 500s, not clean 4xxs (PB-07)

Reproduced directly against a live `TestClient`, all four:

  - POSTing the same `external_ref` twice to `/score` raised an unhandled
    `sqlalchemy.exc.IntegrityError` (`UNIQUE constraint failed:
    claims.external_ref`) -> bare 500.
  - Submitting feedback twice for the same `claim_id` to `/feedback`
    raised the same kind of unhandled `IntegrityError` (unique constraint
    on `investigator_feedback.claim_id`) -> 500.
  - Uploading a genuinely empty file to `/score/batch` raised
    `pandas.errors.EmptyDataError: No columns to parse from file` -> 500.
  - Uploading a header-only CSV (valid header, 0 data rows) got past
    `pd.read_csv` but then raised sklearn's `ValueError: Found array with
    0 sample(s) ... while a minimum of 1 is required by StandardScaler`
    deep inside `_prepare()` -> 500.
  - A fifth, related silent-failure case (not a 500, but the same class
    of "expected mistake handled badly"): a non-numeric value in a
    numeric CSV column (`age="thirty"`) was silently accepted and scored
    — `feature_engineering.py`'s `pd.to_numeric(...,
    errors="coerce").fillna(0.0)` turned it into `age=0.0` with no error
    at all, the batch-CSV analogue of the bug PB-06 fixed for the
    single-claim JSON endpoint.

**Fix.** `api/scoring.py`'s `/score` now queries for an existing
`external_ref` before inserting and returns a 409 with the conflicting
`claim_id` if found, instead of relying on the database to reject it.
`api/feedback.py`'s `/feedback` does the same pre-insert existence check
for `claim_id`, returning 409. `/score/batch` now: catches
`pd.errors.EmptyDataError`/`ParserError` from `pd.read_csv` and returns
400; explicitly checks `len(df) == 0` after a successful parse (the
header-only case, which parses fine but has no rows) and returns 400; and
runs a new `_validate_numeric_columns()` check against
`feature_engineering.NUMERIC_PASSTHROUGH_COLUMNS` before scoring, raising
a 422 listing exactly which columns (and up to 10 example row indices)
failed to parse as numbers, rather than silently corrupting them to 0.0.

`backend/tests/test_client_error_handling.py` (new) locks in all five
fixes: duplicate external_ref -> 409, duplicate feedback -> 409, empty
CSV -> 400, header-only CSV -> 400, non-numeric CSV value -> 422 (with
the offending columns named), and confirms a genuinely valid CSV still
scores successfully end-to-end.

Tests: 45/45 backend (39 existing + 6 new), 3/3 dashboard. No retrain
needed — this ticket only touches API error handling.

### 15. Invalid PSI/drift comparison — scaled vs. unscaled, then a masked boolean-dtype crash (PB-10)

Two bugs, found back-to-back: fixing the first one exposed the second.

**Bug 1 — scale mismatch.** `evaluate_oracle.py`'s Oracle-vs-internal PSI
comparison read `risk_scores_test.csv` as the "reference" distribution and
compared it against `X_oracle`'s engineered features via `psi_report()`.
`risk_scores_test.csv`'s feature columns are `X_test_scaled` —
StandardScaler output (z-scores, mean~0/std~1) — while `X_oracle` is raw,
unscaled engineered features. Comparing a z-scored reference against an
unscaled comparison is an apples-to-oranges scale mismatch: PSI's
quantile bucket edges are computed from the reference distribution, so
buckets built from a mean-0/std-1 reference put nearly all of the
unscaled comparison's real-world-range values into the extreme outlier
buckets, inflating PSI to meaningless numbers.

Reproduced directly (`age`, scaled reference vs. unscaled comparison):

```
risk_scores_test.csv['age']  (X_test_scaled): mean=-0.037, std=1.018, range [-2.20, 2.52]
X_oracle['age']  (raw, unscaled):              mean=40.67, std=12.18, range [16, 80]
psi_numeric(scaled reference, unscaled comparison) = 6.919010289567263
```

A PSI of 6.9 is nonsensical — PSI is conventionally read on a roughly
0-2 scale, and anything >= 0.2 is already "significant drift". The fair,
apples-to-apples comparison (raw `age` from `insurance_claims_cleaned.csv`
as reference vs. raw `X_oracle['age']`) gives:

```
psi_numeric(unscaled reference, unscaled comparison) = 0.047929208048615723   -> "no significant shift"
```

**Fix.** `train.py` now writes a second, dedicated artifact,
`data/processed/psi_reference_features.csv` — `X_test` (unscaled, not
`X_test_scaled`) plus `y_true` — immediately alongside the existing
`risk_scores_test.csv` write. `risk_scores_test.csv` itself is untouched
(the dashboard's `/monitoring/drift` endpoint still legitimately reads it,
but only compares `y_proba`, an unscaled 0-1 probability on both sides —
no bug there, confirmed by reading `api/monitoring.py`). `evaluate_oracle.py`
now reads `psi_reference_features.csv` instead of `risk_scores_test.csv`
for its `internal_test` reference frame, so both sides of the PSI
comparison are raw engineered feature values.

**Bug 2 — boolean-dtype crash, found while validating the fix above.**
Re-running `python -m app.ml.evaluate_oracle` after the Bug 1 fix crashed:

```
TypeError: numpy boolean subtract, the `-` operator, is not supported, use the
bitwise_xor, the `^` operator, or the logical_xor function instead
```

Traceback: `psi.py`'s `_bucket_edges()` -> `reference.quantile(quantiles)`
-> pandas' `Block.quantile()` -> numpy's `_lerp()` -> `b - a` on a boolean
array. Root cause: `psi_reference_features.csv`'s one-hot columns are
bool dtype — pandas 3.0.2's `pd.get_dummies()` now emits bool (not the
historical uint8/int8), and `to_csv()`/`read_csv()` round-trips bool
columns as `"True"`/`"False"` text, correctly re-inferred back to bool
dtype on read. `pd.api.types.is_numeric_dtype()` returns `True` for bool
Series (bool is a numeric subtype in pandas' type system), so
`psi_report()` was routing bool columns into `psi_numeric()`, which calls
`.quantile()` — and numpy's quantile interpolation cannot subtract
booleans.

This bug was **latent, not new**: it existed before Bug 1's fix too, but
was accidentally masked, because the old (buggy) reference,
`risk_scores_test.csv`, is StandardScaler output — every column is
float64 by construction, so `.quantile()` was never called on a bool
Series. Fixing the scale mismatch by switching to
`psi_reference_features.csv` (which does have real bool-dtype one-hot
columns) is what surfaced it.

**Fix.** `psi.py`'s `psi_report()` now excludes bool dtype from its
numeric-vs-categorical routing check (`is_numeric_dtype(...) and not
is_bool_dtype(...)`), sending bool columns to `psi_categorical()`
instead. This is also the semantically correct choice independent of the
crash: a 0/1 one-hot flag is fundamentally a category, not a continuous
quantity to bucket into quantiles.

**Verification.** Re-ran `python -m app.ml.evaluate_oracle` end-to-end
after both fixes: no crash, and `age`'s PSI came out `0.048908` (matches
the 0.048 hand-reproduction above almost exactly, `significant_drift:
false`) — the honest number, no longer the 6.9 artifact. Other
`ORACLE_REAL_FIELDS` (e.g. `auto_year`, `policy_deductable`, `witnesses`)
still show high PSI (6-8) — that's a genuine population difference
between the two datasets on those fields, not a scale-mismatch artifact,
and is expected: it is the same "different population" story already
documented for the Oracle external-validation collapse elsewhere in this
file, not something this ticket is meant to fix.

`backend/tests/test_psi.py` (new) locks in both fixes: `psi_numeric` on
matched vs. shifted synthetic distributions; `psi_report()` no longer
crashes on boolean columns and routes them to `psi_categorical()`
(checked against calling `psi_categorical()` directly); and, when
`psi_reference_features.csv` exists, that its `age` column is on a raw
human scale (not z-scored) and that its one-hot columns really are bool
dtype (the exact fixture this bug lives in).

Tests: 51/51 backend (45 existing + 6 new), 3/3 dashboard. Retrained
(`python -m app.ml.train`) to regenerate `psi_reference_features.csv`;
holdout ROC-AUC unchanged at 0.849439 as expected (PB-10 touches no
features or model code).

### 16. Local SHAP explanations — wrong explainer for the model type, too few reasons, not plain-language (PB-05)

Deferred out of PB-04's scope at the time (`explainer.py`'s `top_reasons()`
hardcoded `k=3` in every caller). Two problems, addressed together.

**Problem 1 — `ClaimExplainer` assumed every model is a tree ensemble.**
`explainer.py` unconditionally built `shap.TreeExplainer(model)` regardless
of what `model` actually was. This project trains and saves three models
(`models/random_forest_final.pkl`, `logistic_regression_final.pkl`,
`xgboost_final.pkl`) and, per §11, D4 explicitly allows a future retrain's
own nested-CV evidence to swap the shipped champion away from Random
Forest. `ClaimExplainer` was only ever instantiated with the RF estimator
in practice (`inference.py`, `train.py`), so this was a real, unexercised
bug rather than one that had been verified safe. Reproduced directly
against the actually-shipped Logistic Regression challenger artifact:

```
>>> shap.TreeExplainer(logistic_regression_estimator)
shap.utils._exceptions.InvalidModelError: Model type not yet supported by
TreeExplainer: <class 'sklearn.linear_model._logistic.LogisticRegression'>
```

**Fix.** `ClaimExplainer.__init__` now picks the SHAP explainer class from
the model's own type: `shap.TreeExplainer` for tree ensembles (Random
Forest, XGBoost — exact, no background data needed, unchanged behavior for
the currently-shipped RF champion), `shap.LinearExplainer` for Logistic
Regression (needs a `background_data` sample of scaled features — raises a
clear `ValueError` up front if none is given, rather than failing deep
inside SHAP), and a generic, sampling-based `shap.Explainer` on
`predict_proba` as a last-resort fallback for anything else. Verified the
fix actually resolves the reproduced crash: built a `ClaimExplainer` on
the real shipped LR estimator with a 30-row background sample of
`risk_scores_test.csv` — no crash, finite SHAP values for every row.

**Problem 2 — `top_reasons()` gave too few, too-technical reasons.**
`k=3` (hardcoded at every call site) frequently left out features the
audit tab specifically calls out (`is_highrisk_hobby`, `is_exec_occupation`)
purely because 2-3 slightly-larger-magnitude features crowded them out of
a 3-item list. Each reason was also just a feature name and a raw signed
SHAP float (`"is_highrisk_hobby raised the fraud risk score (+0.175)."`)
— technically correct but not the kind of plain sentence a non-technical
investigator can act on without already knowing what a SHAP value is.

**Fix.** `explainer.DEFAULT_TOP_K = 8` is the new default (`inference.py`'s
`score_one()` no longer hardcodes `k=3`). Each reason dict now carries
`rank`, `display_name` (humanized feature name), `direction`
(`"increased"`/`"decreased"`, in place of `"raised"`/`"lowered"`), and
`impact` (`"strongly"`/`"moderately"`/`"slightly"`, computed relative to
the single largest-magnitude SHAP driver for THAT claim — raw SHAP
magnitude has no fixed, human-meaningful scale on its own, so an absolute
threshold would be meaningless across different claims) — alongside the
existing `feature`/`value`/`shap_value`/`sentence` keys, all wired through
`api/schemas.py`'s `ReasonCode`. Whether a value displays as `yes`/`no` is
decided from the *column's dtype* (bool, from one-hot encoding) or a known
flag-feature name, never guessed from the value itself — a genuine numeric
feature that happens to equal 0 or 1 (`witnesses=1`) must never be
mislabeled "yes"/"no" just because it looks boolean-ish. Example, live
output for a real claim (`insured_hobbies="chess"`, `witnesses=0`):

```
1. is highrisk hobby (yes) strongly increased the fraud risk score.
2. is major damage (yes) moderately increased the fraud risk score.
3. incident severity ordinal (2) slightly increased the fraud risk score.
4. witnesses (0) slightly decreased the fraud risk score.
```

`backend/tests/test_explainer.py` (new) locks in both fixes: SHAP's
`TreeExplainer` genuinely cannot explain a `LogisticRegression` directly
(reproduction lock-in); `ClaimExplainer` routes RF to `TreeExplainer` and
LR (given `background_data`) to `LinearExplainer` without crashing, and
raises `ValueError` for LR with no `background_data`; `top_reasons()`
defaults to `k=8`, ranks are `1..k` in strict SHAP-magnitude order,
`direction`/`impact` are always one of the expected words,
`display_name` is used inside `sentence`; raw (unscaled) values are used
for display when given; `_format_value`/`_impact_label` unit-tested
directly, including the "numeric 0/1 is never relabeled yes/no" guard.
`backend/tests/test_api.py::test_score_and_retrieve_claim` updated for the
new `k=8` default and the new `ReasonCode` fields.

Tests: 60/60 backend (51 existing + 9 new), 3/3 dashboard. Retrained
(`python -m app.ml.train`) — `ClaimExplainer`'s constructor signature is
backward-compatible for the RF path (no `background_data` needed), so this
is a no-op verification, not a required retrain; holdout ROC-AUC unchanged
at 0.849439.

### 17. Proxy features gated behind a config flag, OFF by default (SH-02 / D3)

`is_highrisk_hobby`/`is_exec_occupation` (`feature_engineering.RISKY_FEATURE_COLUMNS`)
were previously always computed and always shipped — "flagged in the docs
and the dashboard's audit tab" was the entire mitigation. D3's decision
default is to gate them behind a config flag instead, defaulting OFF, and
to measure (not assert) what excluding them costs.

**Fix.** `app/core/config.py` adds `INCLUDE_PROXY_FEATURES`
(env var `INCLUDE_PROXY_FEATURES`, default `false`).
`feature_engineering.engineer_features()` takes a new
`include_proxy_features: bool | None = None` parameter — `None` (every
existing call site) reads the config default; the two risky columns are
now not built at all when off, not merely present-and-ignored. `train.py`
threads the same parameter through `build_features()` and
`cross_validate_model()` so both proxy-feature variants can be trained
from the same functions the shipped pipeline itself uses (no parallel
implementation to drift out of sync).

**Both variants are measured and reported, every training run** — a new
`run_proxy_feature_ablation()` in `train.py` trains the champion
architecture (RF+SMOTE, identical hyperparameters) on both variants,
reporting a single holdout comparison (threshold 0.5 for both, so the
comparison isn't confounded by two different threshold choices — a raw
feature-inclusion effect, not a threshold-selection effect) AND a full
5-fold refit-per-fold CV (a 2-feature difference on a 200-row holdout
split alone is noisy). Saved to
`data/processed/proxy_feature_ablation.csv`; `models/metrics.json` gets a
new `proxy_features` block recording which variant actually shipped.

**The measured cost of turning them off (this training run):**

| | proxy features ON (72 features) | proxy features OFF (70 features, **shipped**) |
|---|---|---|
| Holdout recall | 0.857 | 0.735 |
| Holdout ROC-AUC | 0.849 | 0.794 |
| Holdout PR-AUC | 0.576 | 0.545 |
| Holdout F1 | 0.730 | 0.679 |
| 5-fold CV recall (mean±SD) | 0.886±0.038 | 0.676±0.054 |
| 5-fold CV ROC-AUC (mean±SD) | 0.857±0.031 | 0.770±0.021 |

This is a real, non-trivial cost — recall (this project's own top-ranked
metric) drops by roughly 12-21 points depending on which comparison is
read. It is disclosed here, in `docs/LIMITATIONS.md`, and on the
dashboard's Model Insights → Feature quality audit tab, rather than
hidden or minimized: `feature_engineering.py`'s own module docstring
already judged these two features "close to proxy-discrimination
(lifestyle -> risk score) real insurance regulators scrutinize" — this
project's own conclusion is that shipping a model that leans on an
unexplained lifestyle/occupation -> risk association is not worth that
performance gain, so the honest trade-off is accepted, not hidden behind
a headline number. `INCLUDE_PROXY_FEATURES=true` remains available to
reproduce the ON variant (e.g. for a side-by-side defense discussion).

**A note on the lineage constraint.** This rebuild's own working brief set
a lineage bar of "must not be worse than Project A" (holdout ROC-AUC
0.860, PR-AUC 0.618, recall 0.776 @ threshold 0.55). With proxy features
on, Aegis already met or approached that bar (ROC-AUC 0.849, recall
0.857 at the chosen threshold). With the SH-02/D3 default now OFF, the
shipped model's holdout numbers (ROC-AUC 0.794, PR-AUC 0.545, recall
0.735) fall BELOW that bar on every one of those three metrics. This is
a direct, acknowledged consequence of this ticket's own instruction (D3:
gate proxy features, OFF by default) and is flagged here explicitly
rather than left for a reader to notice on their own: the two constraints
("beat Project A" and "exclude proxy-discrimination-adjacent features by
default") are in real tension for this specific 1,000-row dataset, where
those two features happen to carry a large, genuine share of the
model's signal. `INCLUDE_PROXY_FEATURES=true` recovers the
lineage-beating numbers if that trade-off is preferred; the default stays
OFF because this rebuild reads the two constraints as being in that
order of priority when they conflict, but this is a judgment call worth
surfacing in a defense, not a settled fact.

**Downstream consistency.** `evaluate_oracle.py`'s `engineer_features(mapped_df)`
call (no explicit override) automatically tracks whichever variant is
configured, so the Oracle adapter's feature set always matches
`feature_columns.json` — no separate flag to keep in sync.
`dashboard/app_pages/model_insights.py`'s audit tab text and
`README.md`/`docs/LIMITATIONS.md` were updated to describe "excluded by
default, opt-in" rather than "flagged but kept, removable in one line."

`backend/tests/test_proxy_features.py` (new) locks in: the config default
is `False`; `engineer_features()` with no override, with an explicit
`True`, and with an explicit `False` all produce the correct column set;
the shipped `models/feature_columns.json` genuinely excludes the risky
columns; `metrics.json` discloses the setting; the ablation report exists
with both variants and sane (0-1, finite) metrics for each.

Tests: 68/68 backend (60 existing + 8 new), 3/3 dashboard. Retrained —
this IS a required retrain (the shipped feature set changed from 72 to 70
columns); see the before/after table above for the full metric impact.

### 18. Report uncertainty everywhere a number is reported — CV mean±SD already existed, bootstrap 95% CI did not (SH-04)

Every headline metric this project reports (`model_comparison.csv`'s
holdout recall/ROC-AUC/etc., `oracle_validation_report.json`'s Oracle
numbers, the Overview page's KPI cards) was a bare point estimate off ONE
fixed split, with no sense of how much sampling noise sits under it.
`cross_validate_model()`'s `*_mean`/`*_std` already answered a
complementary question (how much a number moves across different
TRAINING splits) but nothing answered "how much would THIS split's number
move on a different SAMPLE of the same rows."

**Fix.** New `backend/app/ml/uncertainty.py`: `bootstrap_metric_ci()`
resamples `(y_true, y_proba)` row pairs with replacement (1,000
iterations), recomputes ROC-AUC/PR-AUC always and recall/precision/F1/
accuracy when a threshold is given, and returns the empirical 95%
percentile interval per metric — standard, model-agnostic, and doesn't
require retraining anything (it only needs already-computed predicted
probabilities). A resample containing only one class can't score ROC-AUC/
PR-AUC (undefined); those resamples are skipped for those two metrics
only, and `n_boot_effective` discloses how many resamples actually
contributed so a reader can judge how much a given interval should be
trusted (matters most for Oracle's ~6% fraud rate).

Wired into both places a holdout point estimate gets reported:
- `train.py`: after `model_comparison.csv` is written, bootstrap CI is
  computed for all three models (RF/LR/XGB) at their respective
  comparison threshold and saved to
  `data/processed/holdout_bootstrap_ci.csv` (long format: model, metric,
  point_estimate, ci_lower, ci_upper, n_boot_effective).
  `metrics.json` gets a new `uncertainty` block pointing at both the CV
  file and this new one.
- `evaluate_oracle.py`: `oracle_validation_report.json` gets a new
  `oracle_metrics_ci` key, keyed the same as `oracle_metrics`. At the
  current operating threshold the shipped model flags nothing on Oracle
  (recall/precision/F1 all exactly 0 in every one of 1,000 resamples —
  `ci_lower`/`ci_upper` both `0.0`), which is itself informative: the
  collapse isn't a borderline case with a wide interval, it's a flat
  floor.

**Dashboard.** Overview page: each KPI card's caption now shows the
bootstrap 95% CI alongside its point estimate (e.g. "95% CI [0.709,
0.870]"), a new caption line surfaces the RF row's CV mean±SD explicitly
labeled as a different question ("moves across training splits" vs.
"moves across samples of this one test set"), the Oracle warning banner
shows Oracle's own ROC-AUC CI, and the model-comparison table gains
`recall_95ci`/`pr_auc_95ci`/`roc_auc_95ci` columns. Model insights page's
existing CV tab gets a caption explicitly distinguishing the two kinds of
uncertainty so neither is mistaken for "the same interval, computed
twice." `components/data_access.py` gets `load_bootstrap_ci()` +
`bootstrap_ci_for()`.

`backend/tests/test_uncertainty.py` (new): bootstrap CI brackets the real
point estimate for ROC-AUC/PR-AUC on a synthetic classifier; threshold
`None` returns only AUC-family metrics, a given threshold adds the
classification metrics too; determinism given a fixed `random_state`;
empty input returns `{}` without crashing; an all-one-class degenerate
sample correctly omits ROC-AUC/PR-AUC (undefined on every resample) while
still reporting accuracy; `holdout_bootstrap_ci.csv` brackets
`model_comparison.csv`'s point estimates; `metrics.json` discloses both
uncertainty artifacts. `dashboard/tests/test_dashboard_pages.py` gains a
Model Insights smoke test and an Overview-page assertion that the
bootstrap-CI/CV-mean±SD captions actually render (via `streamlit.testing`'s
`AppTest`, not just an import check).

Tests: 76/76 backend (68 existing + 8 new), 5/5 dashboard (3 existing + 2
new). Retrained + re-ran `evaluate_oracle.py` to regenerate the new
artifacts; no feature/model changes, so the point estimates are unchanged
from §17 — only the new CI columns/keys are new.

### 19. Kafka removed entirely, not left unwired (PB-08/PB-09, D1)

The working brief's D1 decision default: remove Kafka entirely rather
than keep a "logic verified directly, live-broker wiring unverified"
disclosure. The reasoning holds up on inspection — no live Kafka broker
was ever reachable in any sandbox this project was assembled in (Docker
Hub network-blocked), so the `confluent_kafka.Producer`/`Consumer` client
wiring in `app/kafka/producer_sim.py`'s own docstring was never actually
exercised against anything, ever, in this project's history. Keeping an
entire subsystem in a delivered project that has literally never run
end-to-end understates its own risk more than the previous disclosure
language admitted.

**Removed entirely** (not stubbed, not disabled behind a flag):
- `backend/app/kafka/` (`producer_sim.py`, `__init__.py`) — the
  `InMemoryKafkaStub`/`validate_claim_message`/`produce_claim`/
  `consume_and_process` logic.
- `backend/app/api/ingestion.py` — the entire file was the two Kafka
  endpoints (`POST /ingest/kafka/produce`, `POST /ingest/kafka/consume`);
  nothing non-Kafka was in it. `main.py`'s router registration for it
  removed too.
- `backend/tests/test_kafka_logic.py` (the in-memory-stub unit tests) and
  `test_api.py::test_kafka_ingest_flow` (the end-to-end route test).
- `KAFKA_BOOTSTRAP_SERVERS`/`CLAIMS_TOPIC` from `app/core/config.py`.
- `confluent-kafka` from `backend/requirements.txt`.
- The `kafka` service and the `api` service's `KAFKA_BOOTSTRAP_SERVERS`
  env var from `docker-compose.yml` (`docker compose config` re-validated
  clean after the edit — Postgres and the app's own two images are now
  the only services).

**What's unchanged.** Claims still enter via `/score` (single JSON) and
`/score/batch` (CSV upload) — `api/scoring.py`, fully tested against a
live `TestClient`, untouched by this ticket. `Claim.ingested_via`'s
column stays (only its allowed-values comment dropped `kafka`) — a
pre-existing DB with historical `"kafka"` rows is not migrated or
rewritten, this is a code change, not a data migration.

**Locked in, not just removed.** `test_api.py` gained
`test_kafka_ingestion_routes_are_gone`, asserting both former Kafka
routes now 404 — a removal is only actually verified once something
checks it stayed removed, the same standard this project holds every
other fix to.

`README.md`/`docs/LIMITATIONS.md` updated from "three ingestion paths" /
"Kafka consumer is at-least-once" to reflect the removal (LIMITATIONS.md
strikes the old bullet through and marks it resolved, per this file's own
established convention, rather than deleting the historical record).

Tests: 71/71 backend (5 fewer than before this ticket — `test_kafka_logic.py`'s
5 tests removed, `test_api.py`'s 1 Kafka test replaced with 1
route-is-gone test — net -5, not a regression), 5/5 dashboard (dashboard
never referenced Kafka, unaffected). No retrain needed — this ticket only
removes an unused ingestion path, no ML code touched.

### 20. API-key auth + PII masking (PB-11)

Reproduced directly: a fresh, completely unauthenticated `TestClient` —
no headers at all — could hit every endpoint, including `GET /claims`,
which returned every stored claim's full `raw_payload` verbatim,
including `insured_zip`. This schema has no name/SSN/DOB field at all,
but a full ZIP is itself a quasi-identifier (HIPAA's Safe Harbor rule
treats it the same way, alongside age/sex — both of which this payload
also carries) that none of this API's actual use cases need in a
response body: scoring only ever needs the real value in-process.

**Fix — auth.** New `app/core/security.py`: `require_api_key()`, a
FastAPI dependency built on `fastapi.security.APIKeyHeader` (so it shows
up as a proper "Authorize" button in `/docs`, not an undocumented
header), checked with `hmac.compare_digest` (avoids a timing
side-channel on the comparison — a real, if minor, thing to get right
even for a single shared secret). Wired centrally in `main.py` via
`app.include_router(..., dependencies=[Depends(require_api_key)])` for
every business router (scoring, claims, feedback, audit, monitoring) —
one place to audit, not N, and nothing added later can accidentally skip
it. `/health` and the auto-generated docs routes stay open (standard
practice — a health check needs to be reachable before any credential is
provisioned). The key itself: `app/core/config.py`'s `API_KEY`, from
`AEGIS_API_KEY`, falling back to a deliberately obvious placeholder
(`CHANGE-ME-insecure-default-api-key`) — this makes "nobody set a real
key" a visible, grep-able fact about a deployment rather than a
silently-working default that looks secure but isn't.
`docker-compose.yml`'s `api` service now sets it from
`${AEGIS_API_KEY:-CHANGE-ME-insecure-default-api-key}` (`docker compose
config` re-validated clean).

**Fix — PII masking.** `mask_pii()` (same module) redacts `insured_zip`
(e.g. `468000` -> `"46XXXX"`), applied at the response boundary only in
`api/claims.py`'s `GET /claims` and `GET /claims/{id}` — what's stored in
the DB and what scoring reads from the request payload are both
untouched, and the function returns a copy, never mutates its input.
`api/feedback.py`'s `GET /feedback/export` deliberately does NOT mask
it — that endpoint's entire purpose is producing a CSV meant to be
appended straight into the real training data (`feature_engineering.py`
reads `insured_zip` from `RAW_FEATURE_COLUMNS` directly, even though no
engineered feature currently derives from it), so masking there would
write a corrupted value into training data for no benefit; what actually
closes the PB-11 gap for that endpoint is that it's no longer reachable
without the same API key as everything else. This scoping is disclosed
explicitly in the endpoint's own docstring and in
`docs/LIMITATIONS.md`, not left for a reader to wonder why one endpoint
looks "unfixed."

**What this is not.** A single, deployment-wide shared secret — not
per-user auth, not RBAC, not encryption at rest. Genuinely out of scope
for a thesis-scale project and disclosed as such in
`docs/LIMITATIONS.md`; what this ticket closes is specifically
"reachable with literally zero credential at all."

**Test fixture impact.** Every existing `TestClient`-based test file
(`test_api.py`, `test_input_validation.py`, `test_client_error_handling.py`)
now sets `AEGIS_API_KEY` via `monkeypatch.setenv()` and constructs its
`TestClient` with a matching `X-API-Key` header by default, so those
files keep exercising the endpoint logic they were originally written
for rather than all failing on 401 — auth itself is covered separately.

`backend/tests/test_auth.py` (new): missing/wrong API key -> 401 on
representative business endpoints across every router; correct key ->
200; `/health`/`/openapi.json` need no key at all; `GET /claims` and
`GET /claims/{id}` both mask `insured_zip`; `mask_pii()` doesn't mutate
its input and handles a missing/`None` zip gracefully; a source-level
guard that `require_api_key()` still uses `hmac.compare_digest` (against
someone "simplifying" it back to `==` later).

Tests: 80/80 backend (71 existing + 9 new), 5/5 dashboard (the dashboard
calls `FraudScoringService` in-process, never over HTTP — confirmed by
inspection, not just assumed — so it's entirely unaffected by this
ticket). No retrain needed — no ML code touched.

### 21. Dashboard scores and escalations never reached the DB (PB-12)

Reproduced directly: score a claim from the dashboard's "Score a claim"
page, then call the API's `GET /claims` — the claim isn't there. It never
was: `score_claim.py` called `FraudScoringService.score_one()` and
rendered the result, full stop; nothing was ever written anywhere. Same
gap, worse, on "Batch review": the scored batch lived only in
`st.session_state["batch_result"]`, gone on the next page refresh, and
"Escalate a claim for investigation" only ever appended to
`st.session_state["escalated_rows"]` — an escalation an analyst clicked
had **no** representation outside that one browser tab's session, not
even a way for a second analyst to see it. A claim scored through the
API showed up in the claims list, the risk grid, and the audit log; the
identical claim scored through the dashboard was invisible everywhere
except the page that had just shown it.

**Fix — shared write path, not two copies of it.** New
`backend/app/db/persistence.py`: `persist_scored_claim()`,
`persist_scored_claims_batch()`, `persist_escalation()` — the
Claim/ScoredClaim/AuditLogEntry write logic that used to live only
inline in `api/scoring.py`'s `/score` endpoint, now centralized so the
dashboard calls the *exact same functions* instead of a second,
inevitably-drifting copy of the same three-table write. This mirrors the
reasoning that already governs scoring itself: `inference.py`'s shared
`FraudScoringService` singleton means the API and dashboard can never
disagree about a fraud *probability*; this module means they can't
disagree about how a score gets *recorded* either.
`api/scoring.py`'s `/score` endpoint was refactored to call
`persist_scored_claim()` (behavior unchanged — same three rows, same
commit point). `/score/batch`'s existing per-row-flush loop was left
untouched deliberately — it has a real, separate efficiency problem
(N+1 inserts) tracked as its own ticket, PB-18, and folding a rewrite of
it into this one would have muddied which fix caused which behavior
change.

**Fix — dashboard gets its own DB session.** The dashboard had never
touched the DB before this ticket, so nothing guaranteed its tables
existed (unlike the API, whose `main.py` lifespan calls `init_db()` on
startup — and `docker-compose.yml`'s `dashboard` service does not
`depends_on` the `api` service, so it can start first). New
`components/data_access.py::get_db_session_factory()`
(`@st.cache_resource` — calls the idempotent `init_db()` once per
process, then returns `SessionLocal`) and `new_db_session()` (a thin
wrapper for "give me a session", `score_claim.py`/`batch_review.py`'s
equivalent of the API's `Depends(get_db)`, just without FastAPI's
request-scoped teardown — each call site commits and closes explicitly
in a `try/finally`, mirroring `get_db()`'s own `finally: db.close()`).

**Fix — three call sites.**
- `score_claim.py`: after `score_one()`, calls `persist_scored_claim(...,
  ingested_via="dashboard")` and now shows a
  `"Saved as claim #N — visible via the API's /claims/{id}..."` caption,
  so scoring one claim from the dashboard is visibly, not just actually,
  the same durable action as scoring it through the API.
- `batch_review.py`: after `score_batch()`, calls
  `persist_scored_claims_batch(..., ingested_via="dashboard_batch")` —
  `score_batch()` returns neither `model_version` (a scalar, same for
  every row — sourced from `service.model_version` like `/score/batch`
  already does) nor `top_reasons` (batch scoring never calls the SHAP
  explainer at all, so this is explicitly `None` per row, not a bug); the
  returned `claim_id`s are inserted as a column into the results table so
  a reviewer can see, and later escalate, a claim by its real DB id.
- Escalation now writes a `claim_escalated` `AuditLogEntry` (via
  `persist_escalation()`) in addition to the existing
  `st.session_state["escalated_rows"]` append (kept for the same-session
  "escalated so far" table + CSV download — a real DB round-trip on every
  keystroke isn't needed for that, only the escalation action itself).
  `persist_escalation()` is deliberately **not**
  `InvestigatorFeedback` — that table is a ground-truth
  `confirmed_fraud` determination that feeds the retraining export
  (`api/feedback.py`); "escalate for investigation" is a lighter-weight
  "a human should look at this" and is recorded as its own audit-log
  event type so the two are never conflated. Escalating now requires an
  investigator name (new text input; falls back to the existing
  `"dashboard_analyst"` placeholder if left blank, same fallback
  `persist_escalation()` already had).
- `db/models.py`'s `ingested_via` comment updated:
  `api | batch_csv | dashboard | dashboard_batch` — the two new values
  make it possible to tell, from the DB alone, which surface a given
  claim actually came through.

**Test isolation note.** `app/core/config.py`'s `DATABASE_URL` default
points at a real file (`<project_root>/aegis.db`), and
`get_db_session_factory()` is `@st.cache_resource` — a process-global
cache. Without isolating it, every dashboard test run would silently
accumulate rows in that real dev DB. New `dashboard/tests/conftest.py`
sets `DATABASE_URL` to a throwaway per-run SQLite file *before* pytest
collects anything, so `app.db.session` never imports the real path in
the first place.

`backend/tests/test_persistence.py` (new, 9 tests): each of the three
`db/persistence.py` functions exercised directly against a throwaway
SQLite session — correct Claim/ScoredClaim/AuditLogEntry rows and field
values, distinct ids across a batch, the batch length-mismatch
`ValueError`, an empty batch being a no-op rather than an error,
escalation producing a `claim_escalated` entry (never an
`InvestigatorFeedback` row), and the blank-name fallback.
`dashboard/tests/test_dashboard_pages.py` gained two `AppTest`-based
end-to-end checks: scoring a claim through the dashboard produces a
queryable `Claim`+`ScoredClaim`+audit entry tagged `ingested_via="dashboard"`;
uploading a batch persists every row with a matching `ScoredClaim`, and
escalating one produces a `claim_escalated` audit entry with the
submitted investigator name as `actor`.

Tests: 89/89 backend (80 existing + 9 new), 7/7 dashboard (5 existing + 2
new). No retrain needed — no ML code touched.

### 22. Docker/deployment config — credentials, ports, train-init (PB-13, D2)

Reproduced directly against the pre-fix `docker-compose.yml`:

- `postgres` published `5432:5432` to the host with the default
  `aegis`/`aegis` credentials, even though every actual consumer
  (`api`, `dashboard`, `train-init`) reaches it over the compose-internal
  network by service name — the host publish bought nothing for this
  topology and needlessly exposed a trivially-guessable-credential DB to
  whatever network the host sits on.
- The same `aegis`/`aegis` credentials were hardcoded in **four**
  separate places — the `postgres` service's own `environment:` block,
  plus `postgresql://aegis:aegis@postgres/aegis` copy-pasted into
  `train-init`, `api`, and `dashboard`'s `DATABASE_URL` — with no
  override mechanism at all, unlike `AEGIS_API_KEY` (PB-11), which
  already used the `${VAR:-default}` pattern. Rotating a password meant
  editing four lines by hand and hoping they stayed in sync.
- `train-init` (runs `clean_data.py` -> `train.py` ->
  `evaluate_oracle.py`) carried both a `DATABASE_URL` env var and
  `depends_on: postgres: condition: service_healthy` — verified by grep
  that all three of those modules have zero references to
  `db`/`DATABASE_URL`/`SessionLocal`/`get_db`. Training was made to wait
  on, and fail alongside, a database it never opens a connection to.
- Despite the project's own D2 decision ("SQLite default DB, Postgres
  optional"), nothing in the repo said so anywhere `docker-compose.yml`
  was actually read — a reader had no way to know Postgres wasn't a hard
  requirement of the DB access layer itself, just this file's own choice
  of reference datastore.

**Fix.** `postgres`'s `ports:` mapping removed entirely — nothing needs
it published for `docker compose up` to work; a GUI client wanting
direct access can still get it via `docker compose port postgres 5432`
or by adding the mapping back locally. `POSTGRES_USER`/
`POSTGRES_PASSWORD`/`POSTGRES_DB` now read from the shell/`.env` (default
`aegis`/`aegis`/`aegis`, same visibly-insecure defaults as before —
D2/PB-11's "make 'nobody configured a real secret' a visible fact"
reasoning applies here too), and `api`/`dashboard`'s `DATABASE_URL` is
now *built from those same three variables*
(`${DATABASE_URL:-postgresql://${POSTGRES_USER:-aegis}:${POSTGRES_PASSWORD:-aegis}@postgres/${POSTGRES_DB:-aegis}}`)
instead of a fourth hand-typed copy — verified with `docker compose
config` that overriding `POSTGRES_PASSWORD` alone updates the
`postgres` service's own env AND both `DATABASE_URL`s identically, and
that setting `DATABASE_URL` directly overrides the whole composed
string (e.g. to a `sqlite:///` path, to skip Postgres entirely — this is
what makes D2's "Postgres optional" actually exercisable from
`docker-compose.yml`, not just true of `config.py` in isolation).
`train-init`'s `DATABASE_URL` env var and `depends_on: postgres` were
both removed — it now only depends on its own image and the two volumes
it writes into, so training is neither delayed by Postgres's healthcheck
nor made to fail if Postgres has any startup issue. New `.env.example`
documents every overridable variable (`AEGIS_API_KEY`, the three
`POSTGRES_*`, and the `DATABASE_URL` full-override escape hatch).
README's Docker Compose section and `docs/LIMITATIONS.md` both updated
to state D2 explicitly rather than leave it implicit.

`backend/tests/test_deployment_config.py` (new, 9 tests, skipped
wherever the `docker` CLI is unavailable — CI's `ubuntu-latest` runner
has it): shells out to `docker compose config --format json` (stdlib
`json` only, no PyYAML dependency) against the real
`docker-compose.yml` and asserts — postgres has no published port;
api/dashboard keep theirs (8000/8501); train-init has neither a
postgres `depends_on` nor a `DATABASE_URL`; a source-level check that
`clean_data.py`/`train.py`/`evaluate_oracle.py` really do never
reference anything DB-related (the claim the train-init test relies
on); `DATABASE_URL` composes correctly from default and
overridden `POSTGRES_*` values without drifting between `api` and
`dashboard`; a full `DATABASE_URL` override takes precedence; and
`.env.example` documents every variable the compose file actually
reads.

Tests: 98/98 backend (89 existing + 9 new), 7/7 dashboard (unaffected —
no Python code touched). No retrain needed — pure deployment-config
change.

### 23. Batch scoring: N+1 inserts, missing top_reasons, unbounded upload (PB-18)

Reproduced against the pre-fix `/score/batch`, three separate problems:

- **N+1 inserts.** The endpoint's persistence loop did `db.add(claim);
  db.flush()` *inside* a Python `for idx, row in df.iterrows()` — one
  flush (a synchronous DB round-trip to assign the autoincrement id)
  per claim, so scoring N claims did N round-trips instead of one.
  Counted directly rather than inferred: monkeypatching `Session.flush`
  to count real calls (`tests/test_persistence.py`) shows the shared
  `persist_scored_claims_batch()` helper does exactly **1** flush for a
  50-row batch, independent of N — the old inline loop would have shown
  50. Negligible against local SQLite (same process, no network), but a
  real, linearly-growing cost against a network-attached Postgres — this
  repo's own `docker-compose.yml` reference deployment (PB-13).
- **No `top_reasons` for batch-scored claims.** `score_batch()` never
  called into SHAP at all — a claim scored via `/score` always got a
  `top_reasons` explanation, the identical claim scored via
  `/score/batch` got none, silently. Not a performance compromise that
  was ever actually necessary: `ClaimExplainer.shap_values_for()` is
  already vectorized across rows for every explainer kind this project
  uses (tree/linear/generic), so explaining N rows costs one SHAP call
  either way — what made per-row explanation *look* expensive was
  calling `top_reasons()` (built for exactly one row) N separate times,
  each paying its own SHAP call-overhead. New
  `explainer.py::top_reasons_batch()` does ONE SHAP call over the whole
  matrix, then loops in plain Python (cheap) to build each row's reason
  list — refactored the existing per-row logic into a shared
  `_reasons_for_row()` helper so `top_reasons()` and
  `top_reasons_batch()` can't drift apart; a test asserts they produce
  byte-identical output. `score_batch()` now returns a `top_reasons`
  column, persisted through the same `persist_scored_claims_batch()`
  path (no more `top_reasons=None` placeholder in either the API's or
  the dashboard's batch-persistence call).
- **No upload size limit.** The endpoint read the whole uploaded file
  into memory (`await file.read()`) and parsed it with no check on byte
  size or resulting row count at any point — a large-enough file could
  drive unbounded memory/parse/score/SHAP/DB-write/response-payload
  cost from a single request. New `config.MAX_BATCH_UPLOAD_BYTES`
  (10 MB default) rejects an oversized upload with 413 *before* pandas
  parses it; `config.MAX_BATCH_ROWS` (5,000 default) catches the case
  where a small file still unpacks into too many rows. Both
  overridable via env var, same pattern as `FP_REVIEW_COST`/
  `AEGIS_API_KEY`.

**Fix — the endpoint itself.** `/score/batch` now calls the shared
`persist_scored_claims_batch()` (`db/persistence.py`, already used by
the dashboard's Batch review page since PB-12) instead of its own
hand-rolled per-row loop — the same "one write path, not two drifting
copies" reasoning that already governs `/score` (PB-12) now covers both
scoring endpoints. `db/models.py`'s now-unused `ScoredClaim`/
`AuditLogEntry` imports were dropped from `scoring.py` (only `Claim`
is still needed there, for the `/score` duplicate-`external_ref` check).

**Fix — the dashboard side-effect.** `batch_review.py`'s persistence
call no longer hardcodes `"top_reasons": None` (it's real now, straight
from `scored.to_dict(orient="records")`) — but the review *table* still
drops that column before display: a list-of-dicts per cell is useful
data to persist and query later, not something a triage grid should
render inline. Per-claim explanations are still available the same way
they already were (Score a claim's SHAP display, or `GET /claims/{id}`).

`backend/tests/test_explainer.py` (+3): `top_reasons_batch()` produces
byte-identical output to calling `top_reasons()` once per row; respects
a custom `k`; uses raw unscaled values when given, same as the
single-row path. `backend/tests/test_persistence.py` (+1): the
flush-count regression above. `backend/tests/test_client_error_handling.py`
(+3): every row from a real `/score/batch` call carries a non-empty
`top_reasons` list and each claim is independently retrievable via
`GET /claims/{id}`; an oversized upload is 413, not silently
read/parsed/truncated; too many rows is 413, not silently scored.

Tests: 105/105 backend (98 existing + 7 new), 7/7 dashboard (unaffected
— `batch_review.py`'s change is a data pass-through, no new page logic
that AppTest wasn't already exercising). No retrain needed — no model
or feature-engineering code touched.

### 24. Dashboard: hardcoded dates, free-text hobby/occupation (PB-19)

Reproduced directly: `score_claim.py` sent `incident_date="2024-06-15"`,
`policy_bind_date="2018-01-01"` for **every** claim scored through the
dashboard, completely independent of `months_as_customer` or anything
else the analyst entered — no UI control over either field existed at
all. `feature_engineering.py` derives two real model features from these
two raw fields (`policy_age_days`, `vehicle_age_at_incident`), so this
wasn't cosmetic: scoring the identical claim (`months_as_customer=6`)
through the always-hardcoded dates vs. through dates actually consistent
with a 6-month-old policy changed `fraud_probability` by ~0.003 — small
on this dataset's feature-importance profile, but a real, silent,
non-zero effect a dashboard-scored claim had that an API-scored claim
(which lets a caller supply real dates) didn't.

`insured_hobbies`/`insured_occupation` were free `st.text_input` fields
with no indication of the schema's actual vocabulary. Checked what
influence they currently have on the shipped model: **none**, by
design — `feature_engineering.py`'s own module docstring documents that
full one-hot encoding of these two columns was deliberately dropped
(an earlier attempt cross-validated at 0.94 ROC-AUC but collapsed to
0.59-0.78 on a genuine holdout split, a high-cardinality/low-row-count
overfitting signature); their only remaining path into the model is the
two flags SH-02/D3 gates behind `INCLUDE_PROXY_FEATURES` (off by
default). Confirmed directly: scoring the same claim with
`insured_hobbies` set to `"reading"`, `"chess"`, and an outright typo
all produced the identical `fraud_probability` under the default config.
So free text here wasn't silently corrupting today's shipped score — but
it's still a real, disclosed-now risk for `INCLUDE_PROXY_FEATURES=true`
(a typo or case-mismatch silently fails the `isin()`/`==` checks
`is_highrisk_hobby`/`is_exec_occupation` are built from), and a
misleading UI either way: a field that looks like it matters and quietly
doesn't is its own kind of dishonesty this project tries not to ship.

**Fix.** Two new `st.date_input` widgets replace the hardcoded literals.
`policy_bind_date` defaults to `months_as_customer` months before today
(not a disconnected constant) but stays freely editable; `incident_date`
defaults to today and enforces `min_value=policy_bind_date`
(`feature_engineering.py`'s own comment already flags incident-before-bind
as a data issue — a negative "policy age" — so the form doesn't let an
analyst build another instance of it). `insured_hobbies`/
`insured_occupation` are now `st.selectbox`es over two new constants,
`feature_engineering.KNOWN_HOBBIES`/`KNOWN_OCCUPATIONS` — the actual
20/14-value vocabulary from the cleaned training data, defined once
(module-level, same pattern as the existing `HIGH_RISK_HOBBIES`/
`SEVERITY_ORDINAL`) so the UI and the model layer's own membership
checks can't drift apart. A new caption discloses, when
`INCLUDE_PROXY_FEATURES` is off, that hobby/occupation currently don't
affect the score — turning the "looks like it matters, doesn't" gap into
an honestly-stated one instead of a silent one.

**A bug introduced and caught while building this fix.** Making
`incident_date`'s `min_value` track `policy_bind_date`'s live value
created a NEW problem: neither `st.date_input` call had an explicit
`key=`, so Streamlit derived each widget's identity partly from its own
call arguments — meaning `incident_date`'s identity changed whenever
`policy_bind_date`'s value did, and Streamlit silently discarded any
already-entered `incident_date` and reset it to `today()`. Reproduced
directly against a bare `AppTest` session: set Incident date, rerun, set
Policy bind date, rerun — Incident date silently reverted. Fixed by
giving both widgets explicit, stable `key=` values so one widget's state
no longer resets when the other's arguments change. Caught by writing
the regression test for the fix itself, before this ever reached a real
user — the kind of thing this project's "reproduce before you fix, then
prove the fix" discipline is specifically for.

`backend/app/ml/feature_engineering.py` gained `KNOWN_HOBBIES`/
`KNOWN_OCCUPATIONS`. `backend/tests/test_ml_core.py` (+1): both constants
checked directly against the real cleaned training data's unique values
(catches drift if the dataset ever changes, rather than trusting a
hand-typed snapshot indefinitely). `dashboard/tests/test_dashboard_pages.py`
(+4): both date fields are real `st.date_input` widgets with a
self-consistent default ordering; hobby/occupation are `st.selectbox`es
over exactly `KNOWN_HOBBIES`/`KNOWN_OCCUPATIONS`; changing the date
widgets away from their defaults changes what's actually persisted (not
a hardcoded string regardless of the form); and the `key=` fix itself —
editing Policy bind date after Incident date was already set must NOT
reset Incident date.

Tests: 106/106 backend (105 existing + 1 new), 11/11 dashboard (7
existing + 4 new). No retrain needed — no feature-engineering
*computation* changed, only two new documented constants and the
dashboard's own input handling.

### 25. Train/serve skew from per-batch medians, and unsafe batch zip/date parsing (PB-20)

Three independent, reproduced bugs, all touching how `/score/batch`
handles values that fail to parse cleanly.

**Train/serve skew.** `feature_engineering.py`'s `engineer_features()`
derives `policy_age_at_incident_days` and `vehicle_age_at_incident` from
`incident_date`/`policy_bind_date`; when either date failed to parse
(`NaT`), both fell back to `.median()` computed from **whichever rows
happened to be in the current call** — `policy_age_days.median() if
policy_age_days.notna().any() else 365` and the equivalent for the
incident year. That means the identical claim, with the identical
malformed date, could score differently depending on what else was in
its batch — a live, correctness-breaking form of train/serve skew, and
a different failure mode from every other fallback in this file (which
are all fixed constants — see `MISSING_COLUMN_DEFAULTS`). Reproduced
directly: engineered features for one row with an unparseable
`incident_date`, computed alone vs. batched alongside two rows with
deliberately extreme policy ages/incident years, differed in both
derived columns, which moved `fraud_probability` by a small but real and
non-zero amount (this was caught by diffing all engineered feature
columns between the two runs, not just the final score — the first fix
attempt covered only `policy_age_at_incident_days` and still left a
~0.0027 skew from `vehicle_age_at_incident`'s own, separate median
fallback).

**Fix.** Both fallbacks now use fixed constants computed ONCE from the
real training data and hardcoded — the same pattern `KNOWN_HOBBIES`
(§24) and `MISSING_COLUMN_DEFAULTS` already establish:
`POLICY_AGE_FALLBACK_DAYS = 4682` (the real median of
`(incident_date - policy_bind_date).dt.days` over the cleaned training
set) and `INCIDENT_YEAR_FALLBACK = 2015` (the real median incident year
— every row in this dataset is dated 2015, verified directly rather than
assumed). Re-verified after the fix: scoring the identical malformed-date
row alone vs. batched now produces byte-identical `fraud_probability`
(`0.7568784687066251` in both cases).

**Unsafe batch zip/date parsing.** Unlike `/score`'s `ClaimPayload`
schema (range-checked `insured_zip`, ISO-format-validated dates),
`/score/batch` accepted any value at all for `insured_zip`/
`incident_date`/`policy_bind_date` — no equivalent validation existed on
that path. Two concrete consequences, both reproduced directly against a
live `TestClient`:
- `insured_zip` silently corrupted in storage whenever any row in the
  batch had a missing zip: pandas upcasts an int column to float64 the
  instant one cell is empty, so a real zip like `468000` was stored (and
  later PII-masked) as `468000.0`. `GET /claims/{id}` showed `'46XXXXXX'`
  (8 characters — wrong) for the batch-scored claim vs. the correct
  `'46XXXX'` (6 characters) for the identical zip scored via `/score`.
  The genuinely-missing zip in that same batch fared worse: pandas'
  float `NaN` round-tripped as the literal string `"nan"` and masked
  into the nonsensical `'naX'` — worse than either the real masked value
  or a clean null.
- A malformed date string in `incident_date`/`policy_bind_date` sailed
  straight through and fed the per-batch-median `NaT` fallback above —
  the realistic way a typo'd or attacker-supplied batch row could
  trigger that skew in practice, since `/score`'s schema rejects a
  malformed date before it ever reaches `engineer_features()`.

**Fix.** `_validate_zip_and_dates()` raises a 422 (same
`{"error": ..., "columns": {col: [row indices]}}` shape PB-07's
`_validate_numeric_columns` already established) for a *present but
invalid* zip (non-numeric or outside `10,000-999,999`) or a *present but
unparseable* `YYYY-MM-DD` date; an absent cell is not an error, matching
this API's "missing means not supplied" semantics everywhere else.
`_clean_zip_column()` then rebuilds `insured_zip` as a plain Python
`int` (present) or `None` (absent) per cell before storage — bypassing
pandas' automatic upcast entirely, so a batch-scored claim's
`raw_payload` is bit-for-bit the same shape a single-claim one is.

**A pre-existing bug found and fixed along the way.**
`_validate_numeric_columns` (PB-07) decided whether a numeric cell was
"present" via `col.astype(str)` compared against `""`/`"nan"` — which
depends on `.astype(str)` turning a missing (`NaN`) cell into the
literal string `"nan"`. Reproduced directly: on this project's pinned
pandas version (3.0.2), `.astype(str)` on `NaN` leaves an actual float
`NaN`, not the string `"nan"` — so `NaN != "nan"` evaluates `True`, and
every genuinely-missing optional numeric cell read as "present",
non-numeric" and 422'd. A batch CSV row with, say, an empty optional
`total_claim_amount` was being rejected outright — a real, live
regression, not a hypothetical, caught only because this ticket needed
to touch the same "present" logic for the new zip/date checks and a
direct reproduction script was run against it first. Fixed with a new
shared `_present_mask()` helper using `col.notna()` directly — no string
round-trip, dtype-agnostic, and not dependent on pandas' own
version-specific stringification behavior. `pd.read_csv(...,
na_values=[""])` (every call site in this file) already guarantees a
genuinely-empty cell becomes real pandas `NaN`, so this is a strictly
more correct replacement, not a behavior trade-off.

`backend/app/ml/feature_engineering.py`: `POLICY_AGE_FALLBACK_DAYS`,
`INCIDENT_YEAR_FALLBACK` added; both fallback sites switched to them.
`backend/app/api/scoring.py`: `_present_mask()` (new, used by
`_validate_numeric_columns`), `_validate_zip_and_dates()` (new),
`_clean_zip_column()` (new), both new functions wired into
`/score/batch` before scoring. `backend/tests/test_ml_core.py` (+2):
both fallback constants checked directly against the real training
data (same drift-guard pattern as `KNOWN_HOBBIES`); engineered features
for a row with a malformed date verified byte-identical scored alone vs.
batched alongside dissimilar rows.
`backend/tests/test_client_error_handling.py` (+5): a batch row with a
genuinely-missing optional numeric field scores 200, not 422 (the
`_present_mask` fix); an invalid `insured_zip` and an invalid
`incident_date` each 422 with the correct row indices; a genuinely
*absent* zip/date is not an error; and an end-to-end test scoring the
same zip once via `/score` and once via `/score/batch` (mixed in a batch
with a missing zip in the other row, to specifically exercise pandas'
whole-column upcast) confirms both mask identically to `'46XXXX'` while
the missing zip round-trips as a real `null`, not `'naX'`.

Tests: 113/113 backend (108 existing + 2 `test_ml_core.py` + 5
`test_client_error_handling.py`), 11/11 dashboard (unaffected — no
dashboard code touched). No retrain needed — both fallback constants
were computed from the same training data the shipped model was already
fit against; this fixes a scoring-time bug in how a bad/missing input is
handled, not the model itself.

### 26. Negative-path and regression test coverage gaps (PB-16)

Four genuinely untested areas, found by auditing which modules had zero
corresponding test file (not by guessing — `grep`ing every `app/ml/*.py`
and `app/api/*.py` module name against `tests/`'s existing imports):

**`oracle_adapter.py` — this project's sole external-validation
mechanism — had no test file at all.** Its own module docstring makes
specific, checkable claims (exactly 9 of 35 raw fields mapped for real,
by name; every unmapped field left genuinely absent, never invented).
New `tests/test_oracle_adapter.py` (8 tests) checks those claims
directly against the real `data/external/oracle/fraud_oracle.csv`
already in the repo: the mapped-column set matches the documented list
exactly; unmapped fields are genuinely absent (not silently defaulted
inside the adapter itself); `WitnessPresent`'s lossy Yes/No→1/0 mapping
round-trips correctly against the raw source column; `insured_sex`/
`police_report_available` are uppercased to match this project's own
schema vocabulary (a case mismatch here would silently fail every
downstream exact-string categorical check); Oracle's own 0-value
missing-age sentinel (320 of 15,420 rows) is replaced, not passed
through as a literal age of 0; every mapped-and-defaulted row flows
through `engineer_features()` with zero NaNs; and `incident_severity`-
derived features are provably constant across every Oracle row (the
mechanism behind the 93.2%-of-SHAP-weight-constant-on-Oracle finding in
`docs/insurance_fyp_project_handoff.md`, not just its downstream
ROC-AUC symptom).

**`train.py` — every CV/threshold number in `docs/REBUILD_NOTES.md` and
`models/metrics.json` ultimately comes from this file's functions, none
of which had a test.** New `tests/test_train_pipeline.py` (6 tests, all
against the real cleaned dataset, using a fast `LogisticRegression` in
place of the shipped RF+SMOTE pipeline where only the CV/OOF *machinery*
is under test, not the shipped model's own reported numbers — those stay
sourced from an actual training run per this project's own working
rules): `load_and_split()` is deterministic and stays stratified across
repeated calls; `build_features()` aligns test columns onto train
columns with zero NaNs; `_f1_optimal_threshold()` recovers the correct
threshold on a synthetic perfectly-separable grid; `_out_of_fold_proba()`
returns one real (non-degenerate) probability per row; and
`cross_validate_model()` returns valid-range mean/std for every metric
across 5 folds.

**The train/serve parity test this ticket specifically asked for.**
`inference.py`'s `FraudScoringService._prepare()` and `train.py`'s
`build_features()` independently call the same shared
`engineer_features()`/`align_to_training_columns()` functions — nothing
architecturally *forces* them to agree, they just currently do. The new
`test_train_serve_parity_scoring_matches_shipped_artifacts` test takes
three real rows from `train.py`'s own held-out test split, scores each
one through `FraudScoringService.score_one()` (the exact path `/score`
and the dashboard both call), and independently re-derives the same
rows' scores by calling `build_features()` directly against the SAME
shipped `random_forest_final.pkl`/`standard_scaler.pkl` artifacts.
Verified passing with exact (`abs=1e-9`) agreement on all three rows —
this is the one test in the suite that would catch inference.py's
serving pipeline silently drifting from train.py's training pipeline (a
reordered column, a stale `feature_columns.json`, a scaler swapped for a
differently-fit one) before it ever reached a live-scored claim.

**Negative-path API gaps.** `GET /claims/{id}` for a never-scored id and
`POST /feedback` against a never-scored `claim_id` both already returned
a clean 404 in the existing code (`claims.py`/`feedback.py`) but had no
test locking that in — `test_client_error_handling.py` gains both.
`api/monitoring.py` had zero behavioral coverage at all (only an auth
401 check in `test_auth.py`) — new `tests/test_monitoring.py` (3 tests)
covers `/monitoring/kpis`'s happy path and, more importantly,
`/monitoring/drift`'s documented edge case: below its hardcoded 30-row
minimum live volume, it must return `insufficient_live_volume` rather
than feeding a tiny, statistically meaningless sample into `psi_report()`
and returning a misleadingly precise number — verified both below (1
live row) and at/above (30 rows) that threshold.

Tests: 132/132 backend (113 existing + 8 `test_oracle_adapter.py` + 6
`test_train_pipeline.py` + 2 negative-path additions to
`test_client_error_handling.py` + 3 `test_monitoring.py`; net +19), 11/11
dashboard (unaffected — no dashboard code touched). No retrain, no
production code changed — this ticket is pure test-coverage addition, so
every new test asserts against the CURRENT already-fixed behavior (all
passed on the first run against the existing codebase; none of them
uncovered a new bug to fix).

### 27. Auto-generated metrics doc, Oracle statistical caveat, geographic-transferability disclosure (PB-15 + SH-05 + PB-23)

**PB-15.** This project's own working rules require every number in
`docs/` to come from `models/metrics.json`, never be hand-typed — in
practice, before this ticket, that meant manually re-reading
`metrics.json` before writing each number into prose, with no automated
check that a given piece of prose hadn't drifted from the artifacts it
was supposedly sourced from. Building this ticket found a concrete,
reproduced instance of exactly that drift: §3's historical comparison
table (above) states "Oracle ROC-AUC ~0.48 (random)" — accurate against
the training run it was captured from, but the model has since been
retrained several more times by later tickets (PB-14's champion
comparison, PB-24's encoding fix, among others), and the CURRENT
`models/metrics.json`-backed Oracle bootstrap 95% CI is `[0.443, 0.481]`
— a range that no longer contains 0.5 at all, so "random" is no longer
the most accurate one-word characterization (it's now measurably *below*
chance, not merely indistinguishable from it). §3's table has been
annotated to flag itself as a fixed historical snapshot rather than
fixed further, since rewriting old before/after numbers to match today's
model would destroy the historical record of what each ticket actually
changed at the time — instead, `backend/app/ml/generate_metrics_report.py`
(new) is now the canonical, re-runnable source of truth: reads
`models/metrics.json`, `data/processed/cross_validation_results.csv`,
`data/processed/champion_decision.json`, and
`data/external/oracle/oracle_validation_report.json` directly, and writes
`docs/CURRENT_METRICS.md` (also printed to stdout). README.md's "Key
findings" §1 and LIMITATIONS.md's Oracle bullet now point readers at this
script/file instead of asking them to trust a number typed into either
document. `backend/tests/test_generate_metrics_report.py` (5 tests) locks
in that the generated model-comparison table is byte-traceable back to
`metrics.json`, that it writes to `docs/CURRENT_METRICS.md` for real, and
— the specific regression this ticket exists to prevent — that it
correctly reports whether the Oracle ROC-AUC bootstrap CI contains 0.5 or
not, rather than defaulting to a stale "random" label.

**SH-05.** A statistical caveat on the Oracle comparison, disclosed in
both `docs/LIMITATIONS.md` and the generated report itself: Oracle's
15,420 rows give the ROC-AUC comparison real statistical power (the CI
above is tight enough to say with confidence whether it contains 0.5),
but PR-AUC's own baseline shifts with class prevalence — this project's
internal fraud rate (~24.7%) is far higher than Oracle's (~6.0%), so part
of the internal-vs-Oracle PR-AUC gap reflects that prevalence difference
alone, not model degradation. The ROC-AUC comparison and the SHAP
constant-feature-share analysis (both prevalence-independent) remain the
primary, quantified evidence for the generalization failure; the PR-AUC
gap is directionally consistent with that finding but isn't, by itself,
an independently prevalence-controlled confirmation of it.

**PB-23.** A limitation not previously stated anywhere in this repo's
docs, added now: both training datasets (the primary 1,000-row set and
Oracle) are US auto-insurance claims — US dollar amounts, US state codes,
US-specific categorical fields. This project is produced in a University
of Ghana academic context, but unlike the sibling MoMo Guard project
(explicitly grounded in Ghanaian mobile-money data and Bank of Ghana
statistics), no Ghanaian claims data, currency, or regulatory framework
was used, mapped, or validated against anywhere in this pipeline. Stated
explicitly in `docs/LIMITATIONS.md` now, so nothing in this repo's
methodology or reported numbers is mistakenly read as evidence of
transferability to the Ghanaian insurance market — that would need its
own dataset and its own external-validation exercise, the same way the
Oracle adapter answers the (still US-only) generalization question this
project actually does answer.

Tests: 137/137 backend (132 existing + 5 `test_generate_metrics_report.py`),
11/11 dashboard (unaffected). No retrain, no scoring-path code changed —
this ticket adds a docs-generation script plus documentation-only edits.

### 28. Batch-scored claims 500'd on re-retrieval whenever an optional field was blank (PB-26)

Found by live end-to-end testing against a real running server, not by
`pytest` — the exact case this project's "reproduce first" discipline
exists for. Sequence: `POST /score/batch` a 2-row CSV where one row left
an optional field (`total_claim_amount`) genuinely blank — legal and
expected per PB-20's own fix, not an edge case — got a clean `200` back.
`GET /claims/{that claim's id}` immediately afterward returned a bare
`500 Internal Server Error`, with no detail in the response body at all
(`ValueError: Out of range float values are not JSON compliant`, visible
only in the server log). `GET /claims` (the list endpoint) failed the
same way the instant it tried to list ANY claim with this problem.

**Root cause.** A blank CSV cell round-trips through
`pd.read_csv(..., na_values=[""])` as a real Python float NaN.
`api/scoring.py`'s `/score/batch` builds `raw_rows =
df.to_dict(orient="records")` directly from that DataFrame and hands it
to `persist_scored_claims_batch()` (`db/persistence.py`) with no
sanitization at all — nothing between the DataFrame and the
`Claim.raw_payload` JSON column ever converted that NaN to `None`. The
*write* succeeded silently: `json.dumps`'s own default is `allow_nan=True`,
so a literal `NaN` token got written into the stored JSON text without
complaint (confirmed directly by reading the raw SQLite row:
`{"total_claim_amount": NaN, ...}` — not valid RFC 8259 JSON, but Python's
json module doesn't enforce that by default). The *read* is where it
broke: `GET /claims`/`GET /claims/{id}` re-serialize that same payload
through Starlette's default `JSONResponse`, which — unlike `json.dumps`'s
own default — sets `allow_nan=False`, so any claim carrying so much as
one NaN blew up the instant anyone tried to look at it again, even though
scoring itself had already returned a real `200`.

`/score` (single-claim) was never affected: `ClaimPayload.model_dump(exclude_none=True)`
already leaves an unsupplied field genuinely ABSENT from the payload dict
rather than present-with-NaN, so this specific failure mode can't occur
on that path — which is exactly why the existing single-claim-scoring
test suite never caught it, and why this bug is specific to the
batch/pandas path.

**Why the existing test suite missed this.** PB-20's own
`test_batch_csv_with_missing_optional_numeric_field_is_200_not_422`
(added for a different, already-fixed bug) checks only that `/score/batch`
itself returns `200` for a blank optional field — it never follows up
with a `GET /claims/{id}` to confirm the claim it just created can
actually be read back. That's a real coverage gap, not a flaw in that
test's own logic; it just wasn't testing the thing this ticket found.

**Fix.** `db/persistence.py` gains `_sanitize_for_json()` — recursively
replaces any NaN/Infinity float (or a pandas `NaT`, detected the same
dependency-free `x != x` way, with no pandas/numpy import needed) with
`None`, applied to every payload at the ONE shared choke point both
`/score`+`/score/batch` and the dashboard's Score a claim / Batch review
pages already write through (`persist_scored_claim()`/
`persist_scored_claims_batch()` — the same "one write path, not two that
can drift" reasoning PB-12 already established for this module, so this
one fix covers both the API and the dashboard automatically). Re-verified
directly against a live server after the fix: the same reproduction
sequence (blank-field batch row -> `200` -> `GET /claims/{id}`) now
returns `200` with `"total_claim_amount": null`, and `GET /claims`
lists it correctly.

`backend/tests/test_persistence.py` (+5, new `TestSanitizeForJson` class):
unit tests for `_sanitize_for_json()` itself (NaN/Infinity -> `None`,
ordinary values untouched, recurses into nested dicts/lists), plus two
tests that go through the real write functions and a real DB round-trip
— the last one specifically re-serializes the RELOADED row with
`json.dumps(..., allow_nan=False)` (mirroring Starlette's own default,
not `json.dumps`'s permissive one) so it actually fails the way the
original bug did if the fix ever regresses.
`backend/tests/test_client_error_handling.py` (+1): an end-to-end test
through the real FastAPI app — batch-score a row with blank optional
fields, then `GET` both the individual claim and the claims list,
asserting `200` and `null` (not a stringified `"nan"` or a crash) for
every blank field.

Tests: 143/143 backend (137 existing + 6 new across the two files above),
11/11 dashboard (unaffected — no dashboard code changed, though the fix
covers the dashboard's Batch review page equally since it shares
`persist_scored_claims_batch()`). No retrain — this is a persistence-layer
fix, not a scoring or feature-engineering change.
