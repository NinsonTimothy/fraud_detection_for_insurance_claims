# Fraud Risk Scoring — Streamlit Prototype (v2)

## File structure

```
prototype/
├── scoring.py        ← ALL scoring/feature-engineering logic (no Streamlit dependency)
├── app.py             ← Streamlit UI only, imports from scoring.py
├── tests/
│   └── test_prototype.py   ← pytest suite, imports from scoring.py, run headlessly
└── README.md          ← this file
```

**Why split this way:** `scoring.py` has zero Streamlit dependency, so both `app.py` (the UI)
and `tests/test_prototype.py` (automated validation) import the *same* implementation instead
of two copies that could silently drift apart. This also means the test suite runs headlessly
with `pytest`, without launching a browser.

`06_prototype.ipynb` (in your `notebooks/` folder) is a separate, narrated walkthrough of this
same logic, meant for your thesis documentation — it's self-contained rather than importing
`scoring.py`, so it reads standalone if someone opens just the notebook.

## Setup

1. Copy the whole `prototype/` folder (or just `app.py` + `scoring.py` + `tests/`) into a
   location that sits **next to** your existing `models/` and `data/processed/` folders:

   ```
   your-project/
   ├── models/
   │   ├── random_forest_final.pkl
   │   └── shap_explainer.pkl
   ├── data/processed/
   │   ├── X_train.csv
   │   ├── X_test.csv          (needed for the "known cases" test track)
   │   └── y_test.csv          (needed for the "known cases" test track)
   ├── notebooks/
   │   └── 06_prototype.ipynb
   └── prototype/
       ├── scoring.py
       ├── app.py
       ├── tests/test_prototype.py
       └── README.md
   ```

2. Install dependencies:
   ```
   pip install streamlit shap joblib pandas numpy matplotlib scikit-learn xgboost imbalanced-learn pytest
   ```

## Running the prototype (the actual UI)

```
cd prototype
streamlit run app.py
```

Opens a browser tab (usually `http://localhost:8501`) with the claim form.

## Running the tests (headless validation — no browser)

```
cd prototype
pytest tests/ -v
```

- **`TestRiskBanding`, `TestFeatureBuilder`** — pure logic, no model artifacts needed, always run.
- **`TestModelIntegration`** — needs the real `.pkl` files + `X_train.csv`; auto-skips if missing.
- **`TestKnownCasesFromRealTestSet`** — needs `X_test.csv` + `y_test.csv` too; auto-skips if missing.

Run this any time you change `scoring.py` or `app.py` — it catches breakage before you find it
live in the demo.

## What still needs your verification (marked `# TODO-VERIFY` in scoring.py)

I built this from `03_modelling.ipynb` and `05_shap_explanations.ipynb` only — I do not have
`02_preprocessing.ipynb`, so the following are my best inference from column names and SHAP
feature interpretations, not confirmed against your actual preprocessing code:

| Feature | What I assumed | Where to check |
|---|---|---|
| `incident_severity` | Ordinal encoding: Trivial=0, Minor=1, Major=2, Total Loss=3 | Your top SHAP feature is the raw `incident_severity` column (not one-hot), which implies ordinal encoding — confirm the exact mapping |
| `zip3_risk_tier_low_risk` | Placeholder checkbox, not a real ZIP3 lookup | You need the actual fraud-rate-by-ZIP3 table from your EDA notebook to compute this properly |
| `hour_bucket_*` | Night=22:00–05:59, Morning=06:00–11:59, Afternoon=12:00–17:59, Evening=18:00–21:59 | Check your actual bucket boundaries |
| `is_exec_occupation` | Only `exec-managerial` counted | Confirm which occupation categories you flagged |
| `car_age` | `incident_year - auto_year` | Confirm this matches your calculation |
| `policy_age_at_incident_days` | `(incident_date - policy_bind_date).days` | Confirm this matches your calculation |
| `csl_per_person` / `csl_total` | Passed through directly as numbers | Confirm these match how you parsed the CSL string field (e.g. "250/500") |
| One-hot category name matching | e.g. `incident_state_NY`, `authorities_contacted_Police` | If a form selection doesn't match a training column, the app treats it as the reference category and flags it on-screen — worth spot-checking against your actual dummy columns |

**Fastest way to close these gaps:** upload `02_preprocessing.ipynb` (or just the feature
engineering cells) and I'll replace every TODO-VERIFY in `scoring.py` with your exact logic —
`app.py` and the tests will pick up the fix automatically since they both import from it.

## Known limitation carried over from your own notebook

Your Phase 3 limitations section already flags that `zip3_risk_tier` was target-encoded using
global fraud rates rather than recomputed per CV fold — the same caveat applies here even more
directly, since this prototype has no real lookup table at all yet.

