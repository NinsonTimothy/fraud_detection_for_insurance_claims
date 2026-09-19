# Aegis Risk Engine

Explainable insurance-claim fraud scoring — a production-style rebuild of the
DCIT400 final-year project "Explainable Risk Scoring: Insurance Claim Fraud
Detection" (Kwabena Adipah Osei & Timothy Ninson, University of Ghana,
supervised by Prof. Ebenezer Owusu).

Random Forest (SMOTE-balanced) is the shipped fraud classifier, trained on a
real, human-labeled 1,000-row auto-insurance-claims dataset, with SHAP
explainability, a cost-optimal decision threshold, a FastAPI backend (three
ingestion paths: single-claim API, batch CSV, simulated Kafka stream), a
Postgres-backed feature/audit store (SQLite fallback), and a Streamlit
analyst dashboard. It is honestly, disclosedly validated against a second
real dataset (Oracle, 15,420 rows) it never trained on — and that
validation shows the model does **not** generalize past its own training
distribution, with the root cause fully quantified. See
`docs/REBUILD_NOTES.md` and `docs/LIMITATIONS.md` before presenting any
number from this repo — they are part of the deliverable, not an
afterthought.

## Quickstart (no Docker required)

```bash
cd backend
pip install -r requirements.txt
python -m app.ml.clean_data        # -> data/cleaned/insurance_claims_cleaned.csv
python -m app.ml.train             # trains RF/LR/XGB, saves models/, ~30s
python -m app.ml.evaluate_oracle   # external validation + stress test, ~1min
pytest tests/ -q                   # 16 tests, all should pass

uvicorn app.main:app --reload --port 8000   # API at http://localhost:8000/docs
```

In a second terminal:

```bash
cd dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py               # dashboard at http://localhost:8501
```

The dashboard calls the same `FraudScoringService` the API uses, in-process
— they can never disagree about a score.

## Quickstart (Docker Compose)

```bash
docker compose up --build
```

Brings up Postgres, Kafka (KRaft mode, no Zookeeper), a one-shot
`train-init` container (cleans data + trains models + runs Oracle
validation if `models/` is empty), the API (`:8000`), and the dashboard
(`:8501`). **Not independently verified end-to-end** in the sandbox this
was built in — Docker Hub was network-blocked there, so no base image
could be pulled, though `docker compose config` validates the full compose
file. See `docs/LIMITATIONS.md` for the precise disclosure. Budget time to
debug on first real run, the way any un-execute-tested deployment config
deserves.

## Repository structure

```
aegis-risk-engine/
├── backend/
│   ├── app/
│   │   ├── ml/            # clean_data, feature_engineering, train, explainer,
│   │   │                     cost_threshold, psi, inference, oracle_adapter,
│   │   │                     evaluate_oracle
│   │   ├── api/            # scoring, claims, feedback, audit, monitoring, ingestion
│   │   ├── db/              # SQLAlchemy models + session (Postgres/SQLite)
│   │   ├── kafka/            # producer_sim.py — logic verified directly, see LIMITATIONS
│   │   ├── core/              # config
│   │   └── main.py             # FastAPI app
│   ├── tests/                   # 16 tests: ML core (incl. the single-row-scoring
│   │                              regression test), API, Kafka logic
│   └── requirements.txt
├── dashboard/
│   ├── streamlit_app.py           # nav shell
│   ├── app_pages/                 # Overview, Score a claim, Batch review,
│   │                                Model insights, Monitoring & external validation
│   └── components/                 # theme.py, data_access.py
├── data/
│   ├── raw/insurance_claims_raw.csv        # real, 1,000 rows, 24.7% fraud
│   ├── cleaned/                              # + cleaning_log.csv, missing_value_treatment_plan.csv
│   ├── processed/                             # model_comparison.csv, cross_validation_results.csv,
│   │                                            shap_feature_importance.csv, cost_threshold_sweep.csv
│   └── external/oracle/                        # real, 15,420 rows, 6.0% fraud
├── models/                                       # .pkl artifacts, metrics.json, feature_columns.json
├── docs/
│   ├── ml_feature_critique.md                     # original critique (verbatim) this rebuild implements
│   ├── generalization_and_cv_results.md            # original external-validation writeup (verbatim)
│   ├── REBUILD_NOTES.md                             # what changed here vs. the original, and why
│   └── LIMITATIONS.md                                # every known, disclosed limitation
├── deployment/                                        # Dockerfiles, entrypoint-train.sh
├── docker-compose.yml
└── README.md                                            # this file
```

## Key findings to lead with in a defense

1. **Internal performance is honest but modest** (~0.67 ROC-AUC, 5-fold CV,
   full-pipeline-refit-per-fold) — lower than a leaked evaluation would show,
   deliberately, because the leak was found and fixed (`REBUILD_NOTES.md` §2).
2. **The model does not generalize to Oracle** (ROC-AUC ≈0.48, statistically
   random) — but a stress test proves Oracle itself IS learnable fraud data
   (fresh XGBoost reaches ROC-AUC ≈0.81 on Oracle's own fields). This is a
   feature-availability problem, precisely quantified, not a data problem.
3. **Two specific features are flagged as not safe to treat as real signal**
   (`zip3_risk_tier` — disclosed leakage; `is_highrisk_hobby`/`is_exec_occupation`
   — dataset artifacts) and are individually removable in one line each.
4. **What real SIU tooling has that this doesn't** (prior-claims history,
   fault attribution, network-link analysis) is named explicitly as an
   architectural ceiling this dataset cannot support — not glossed over.
