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
