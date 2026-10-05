# Generalization and cross-validation results

> **Historical document (Version A audit).** Its numbers describe the ORIGINAL model. For the current
> model, the Oracle ROC-AUC confidence interval lies entirely below 0.5 — a **significantly inverted**
> ranking, not a random one. Current figures: `docs/CURRENT_METRICS.md`.

> **Note on this copy:** this is the original external-validation writeup
> for the FYP prototype, carried into this repo verbatim as the source
> document this rebuild implements against. The methodology it describes
> (honest field-mapping adapter, fresh-model stress test) is exactly what
> `backend/app/ml/oracle_adapter.py` and `backend/app/ml/evaluate_oracle.py`
> implement in this rebuild. See `docs/REBUILD_NOTES.md` for the actual
> numbers this rebuild produced and why they differ slightly (short
> version: a CV-leakage bug found and fixed here lowers the internal CV
> number; the Oracle collapse and the stress-test recovery both replicate
> closely).

This document answers the question the project had not yet asked of itself: does
`random_forest_final.pkl` — the model actually served by the Streamlit prototype —
generalize past the 1,000-row dataset it was trained on? It has two parts: (1) proper
stratified cross-validation on the training data itself, to put honest uncertainty bounds
on the single train/test split reported in `03_modelling.ipynb`; and (2) a genuine
out-of-sample external validation against `data/raw/fraud_oracle.csv`, a real, independently
collected 15,420-row dataset already sitting unused in the repo. Notebook
`04_pipeline_stress_test.ipynb` used this same Oracle file, but only to train **fresh**
`oracle_*` models on it — its own Section 4.7 says outright *"this is a stress test, not
an independent validation."* This is the independent validation that was still missing.

Everything below was computed by loading the real `.pkl` artifacts and running them
against real data — nothing here is estimated or asserted without a number behind it.

---

## 1. Stratified 5-fold cross-validation (internal rigor check)

`03_modelling.ipynb` reports metrics from a single 800/200 train/test split. With only
1,000 labeled rows, a single split is not a reliable estimate of how the model performs
in general — a different random split could easily land 5-10 points of PR-AUC away by
chance alone. Stratified 5-fold CV over the full 1,000 rows (fraud rate 24.7% preserved
in each fold) quantifies that uncertainty directly, using each model's exact saved
pipeline (SMOTE + the tuned classifier — identical hyperparameters to the shipped
`.pkl` files, not re-tuned):

| Model | ROC-AUC | PR-AUC | F1 @ 0.5 | Recall @ 0.5 | Precision @ 0.5 |
|---|---|---|---|---|---|
| Random Forest (shipped model) | 0.877 ± 0.036 | 0.644 ± 0.086 | 0.728 ± 0.055 | 0.805 ± 0.083 | 0.666 ± 0.047 |
| Logistic Regression | 0.885 ± 0.031 | 0.666 ± 0.077 | 0.708 ± 0.052 | 0.793 ± 0.075 | 0.641 ± 0.046 |
| XGBoost | 0.883 ± 0.030 | 0.675 ± 0.072 | 0.750 ± 0.037 | 0.886 ± 0.039 | 0.652 ± 0.044 |

*(F1/Recall/Precision above use each model's default 0.5 threshold, not the tuned
operating thresholds in `model_comparison.csv` (0.55/0.75/0.6) — ROC-AUC/PR-AUC are the
threshold-free numbers to compare directly against that table.)*

**Reading this honestly:** the single-split Random Forest numbers already reported
(ROC-AUC 0.860, PR-AUC 0.618) sit comfortably inside the 5-fold range above — the
original split was not a lucky draw. But the ±0.036 / ±0.086 spread is real and worth
stating in the thesis alongside any single-split number: on 1,000 rows, a claim of
"PR-AUC 0.618" has meaningfully more uncertainty around it than the same number would
on a 100,000-row dataset. XGBoost edges out Random Forest on every CV metric here
(higher PR-AUC, higher recall, comparable ROC-AUC) — worth a line in the thesis on why
Random Forest was still selected as final (interpretability/SHAP stability is a
legitimate reason, but it should be stated, not left implicit).

Reproducible via `prototype/analysis/cross_validate.py` (added to the repo).

---

## 2. External validation: Oracle dataset (real generalization test)

### Method — an honest adapter, not a cherry-picked one

`oracle_adapter.py` (added to `prototype/analysis/`) maps `fraud_oracle.csv`'s columns
onto the 35 raw columns `scoring.engineer_batch_features()` expects, using the exact
same batch-scoring code path the Streamlit app itself uses (no shortcut, no separate
scoring logic). Every field mapped below is a genuine semantic match; everything else is
left absent on purpose so `scoring.py`'s own documented `MISSING_COLUMN_DEFAULTS`
fallback handles it — nothing is invented to make the model look better or worse.

| Trained feature | Oracle source | Fidelity |
|---|---|---|
| `age` | `Age` | Direct |
| `insured_sex` | `Sex` | Direct |
| `policy_deductable` | `Deductible` | Direct (different value range: $300-700 vs $500/1000/2000) |
| `police_report_available` | `PoliceReportFiled` | Direct |
| `witnesses` | `WitnessPresent` | Yes/No → 1/0 (lossy vs. a real count) |
| `auto_make` | `Make` | Direct, but only 7 of 19 Oracle makes match `scoring.py`'s `MAKE_MAP` spelling exactly (e.g. Oracle's "Accura"/"Nisson"/"VW" don't match "Acura"/"Nissan"/"Volkswagen") — 53.7% of rows resolve to a real region, the rest fall to "Other", exactly like the pre-existing Accura/Suburu quirk `scoring.py` already documents |
| `number_of_vehicles_involved` | `NumberOfCars` (bucketed) | Bucket midpoint |
| `incident_date` | `Year` + `MonthClaimed` + `WeekOfMonthClaimed` | Reconstructed real calendar date |
| `auto_year` | `incident_date.year` − `AgeOfVehicle` bucket midpoint | Recovers a real `car_age` |
| `policy_bind_date` | `incident_date` − `Days_Policy_Claim` bucket midpoint | Recovers a real (bucketed) policy-age signal |

10 of 35 raw fields mapped for real; the remaining 25 — including `incident_severity`,
`insured_hobbies`, `insured_occupation`, `insured_zip`/state/city, and every claim-dollar
field (`injury_claim`/`property_claim`/`vehicle_claim`) — have **no Oracle equivalent at
all**, because Oracle simply doesn't collect them.

### Results

| Metric | Internal holdout (trained model) | Oracle external validation |
|---|---|---|
| Rows | 200 | 15,420 |
| Fraud rate | 24.5% | 6.0% |
| ROC-AUC | 0.860 | **0.496** (= random for that Version-A model; the CURRENT model is significantly inverted, see note) |
| PR-AUC | 0.618 | **0.060** (≈ Oracle's own base rate, 0.060) |
| Claims flagged at the trained 0.55 threshold | 77.6% recall | **0** — every probability the model outputs on Oracle is between 0.271 and 0.318; nothing ever crosses 0.55 |

ROC-AUC of 0.496 means the model's ranking of Oracle claims is statistically
indistinguishable from random. This is a harder collapse than the sibling mobile-money
project's PaySim result (which at least retained ROC-AUC 0.71, better than random) — here
there is no useful signal left at all once the model leaves its training distribution.

### Root cause — quantified the same way as the sibling project's PaySim analysis

Cross-referencing `data/processed/shap_feature_importance.csv` (the model's own global
SHAP weights) against which of the 94 trained columns are actually **constant** once
Oracle data is mapped through the real adapter above:

| | Feature count | Share of total SHAP weight |
|---|---|---|
| Constant on Oracle-mapped data (no discriminating signal) | 81 of 94 | **93.2%** |
| Genuinely variable on Oracle-mapped data | 13 of 94 | 6.8% |

The single most important feature in the entire model, `incident_severity` (17.1% of
total SHAP weight — 3.5× the #2 feature), cannot be computed from Oracle at all and sits
frozen at its neutral default for all 15,420 rows. The next four highest-weight features
(`zip3_risk_tier_low_risk`, `is_major_damage`, `is_highrisk_hobby`,
`authorities_contacted_Police` — together another 14.6%) are equally frozen. Combined,
the top 5 features alone account for 26.6% of total SHAP weight and are 100% constant on
Oracle — before even counting the other 76 constant features. This is not a borderline
case: the model's decision-making is built almost entirely out of fields a real external
transaction source simply does not provide.

**This is a feature-availability problem, not evidence Oracle is unlearnable fraud
data.** `04_pipeline_stress_test.ipynb`'s own fresh-trained models on Oracle prove the
opposite — XGBoost trained directly on Oracle's real fields reaches ROC-AUC 0.827, PR-AUC
0.231 (`data/processed/oracle_model_comparison.csv`). The signal is there; the primary
model was simply never built to use fields that would survive a change of data source.

### What to change — prioritized, following the same logic as the PaySim analysis

1. **Report generalization risk in the thesis honestly, next to the internal numbers.**
   An internal ROC-AUC of 0.86 next to an external ROC-AUC of 0.50 is a stronger, more
   defensible finding for a final-year project than a clean internal number alone — very
   few student fraud-detection projects check this at all.
2. **Stop leaning almost entirely on fields that only exist inside this specific 1,000-row
   dataset's collection process.** `incident_severity`, ZIP-based risk tier, and hobby/
   occupation flags are all dataset-specific or leakage-prone (see
   `ml_feature_critique.md`) — the same features driving the generalization collapse are
   independently flagged there as questionable for production use anyway. Fixing the
   feature set fixes both problems at once.
3. **Feature-dropout / simulated-missingness augmentation during retraining** — the same
   fix recommended in the sibling project: randomly blank out `incident_severity`,
   hobbies, occupation, ZIP3 tier on a fraction of training rows so the model is forced
   to learn a fallback that isn't "assume everything is fine," rather than collapsing
   toward the majority class when its favorite features are absent.
4. **Treat this Oracle validation as a standing regression check.** `prototype/analysis/`
   now has a re-runnable adapter + evaluation script — run it after any future retrain,
   the same way `evaluate_paysim.py` is used as a gate in the sibling project.

---

## Reproducing this

```
cd prototype/analysis
python cross_validate.py        # Section 1
python evaluate_oracle.py       # Section 2
```

Both scripts load the real `.pkl` artifacts under `models/` and write their results to
`prototype/analysis/results/`. Nothing here was hand-typed from memory — every number in
this document was generated by these two scripts.
