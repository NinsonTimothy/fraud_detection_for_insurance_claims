# Aegis Risk Engine

An explainable insurance-claim fraud scoring system developed by **Timothy Ninson** and **Kwabena Adipah Osei**.

Aegis Risk Engine is a rebuilt, production-style fraud detection system for scoring auto-insurance claims. It combines machine-learning classification, SHAP-based explainability, API-based scoring, batch review, persistence, monitoring, external validation, and an analyst dashboard in one application.

The earlier version of the project is preserved under [`legacy/version-a/`](legacy/version-a/).

> **Project status:** Academic/research prototype. It is designed to demonstrate an end-to-end explainable fraud-risk workflow and should not be treated as a production insurance decision engine without further validation, security hardening, governance, and operational controls.

---

## Overview

Aegis Risk Engine supports:

- Single-claim fraud scoring through a FastAPI API
- Batch claim scoring from CSV files
- Leak-free champion selection (LR / RF / XGBoost vs. a one-line rule baseline)
- SHAP-based local and global explanations
- Cost-sensitivity analysis over a grid of stated assumptions
- Claim, audit, and feedback persistence
- Model monitoring and external validation
- Streamlit analyst dashboard
- PostgreSQL support with SQLite fallback
- Docker Compose deployment
- Automated backend testing

The shipped classifier (the *champion*) is chosen automatically by a pre-declared rule on the 800 development rows — currently a calibrated **Logistic Regression**, compared against Random Forest, XGBoost and a one-line rule baseline (see [`docs/CURRENT_METRICS.md`](docs/CURRENT_METRICS.md)). It is trained on a 1,000-row auto-insurance claims dataset and also evaluated against a separate 15,420-row external dataset to test whether its performance transfers beyond the training distribution.

That external validation reveals an important limitation: the model does **not** generalize strongly to the external dataset. This is intentionally documented rather than hidden. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/CURRENT_METRICS.md`](docs/CURRENT_METRICS.md) before interpreting model performance.

---

## Architecture

```text
                    ┌──────────────────────────┐
                    │   Insurance Claim Data   │
                    └────────────┬─────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │ Cleaning & Features   │
                     │   ML preprocessing    │
                     └───────────┬───────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │ Champion-model Risk Scoring│
                  │ + threshold / risk policy  │
                  └──────────────┬──────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ SHAP Explainability     │
                    └────────────┬────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
   ┌────────▼────────┐  ┌────────▼─────────┐  ┌──────▼─────────┐
   │ FastAPI Backend │  │ Streamlit        │  │ PostgreSQL /   │
   │ single + batch  │  │ Analyst Dashboard│  │ SQLite Store   │
   └─────────────────┘  └──────────────────┘  └────────────────┘
```

Claims can be scored individually through the API or in batches from CSV files. The dashboard uses the same scoring service as the backend, keeping model behavior consistent across interfaces.

---

## Tech Stack

### Machine Learning

- Python
- scikit-learn
- XGBoost
- imbalanced-learn (imbalance experiment only; the shipped model uses class weighting)
- SHAP
- pandas
- NumPy

### Backend

- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL
- SQLite

### Dashboard

- Streamlit
- Plotly

### Deployment and Testing

- Docker Compose
- pytest

---

## Repository Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/          # scoring, claims, feedback, audit, monitoring
│   │   ├── core/         # configuration and API-key security
│   │   ├── db/           # SQLAlchemy models and persistence
│   │   └── ml/           # cleaning, training, inference, SHAP, monitoring
│   ├── tests/            # backend and ML tests
│   └── requirements.txt
│
├── dashboard/
│   ├── app_pages/        # analyst-facing dashboard pages
│   ├── components/       # shared dashboard utilities
│   ├── streamlit_app.py
│   └── requirements.txt
│
├── data/
│   ├── raw/              # primary training data
│   ├── cleaned/          # cleaned data and cleaning artifacts
│   ├── processed/        # evaluation and monitoring outputs
│   └── external/         # external validation data
│
├── models/               # trained model artifacts and metrics
├── docs/                 # metrics, limitations, rebuild notes and analysis
├── deployment/           # Dockerfiles and training entrypoint
├── legacy/
│   └── version-a/        # preserved earlier project version
│
├── .env.example
├── docker-compose.yml
└── README.md
```

---

# Getting Started

Aegis can be run in two ways:

1. **Without Docker** — best for development, debugging, testing, and exploring the code.
2. **With Docker Compose** — best for starting the full stack together with PostgreSQL.

## Prerequisites

For a local installation:

- Git
- Python **3.12**
- pip

Python 3.12 is recommended for the pinned dependency set. In particular, the project pins `psycopg2-binary==2.9.9`, which is more straightforward to install under Python 3.12 than Python 3.13 on Windows.

For the containerized installation:

- Git
- Docker Desktop or Docker Engine
- Docker Compose

---

# Option A — Run Without Docker

## 1. Clone the repository

```bash
git clone https://github.com/NinsonTimothy/fraud_detection_for_insurance_claims.git
cd fraud_detection_for_insurance_claims
```

## 2. Create a Python 3.12 virtual environment

### Windows — Git Bash

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

### Windows — PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Confirm the interpreter:

```bash
python --version
```

You should see Python 3.12.x.

## 3. Install dependencies (backend and dashboard install together)

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt -r dashboard/requirements.txt
```

## 4. Regenerate every artifact (one command)

```bash
cd backend
python -m app.ml.run_all
```

This cleans the data, runs the development-only imbalance experiment, trains and selects the champion
(10x5 repeated CV on the 800 development rows, rule baseline included, champion computed by a
pre-declared rule), calibrates, derives the risk bands and the cost-sensitivity grid, evaluates once on
the 200 test rows, runs the Oracle external validation and the sensitivity probe, and regenerates
`docs/CURRENT_METRICS.md`, `docs/THESIS_UPDATE_NOTES.md` and `docs/VIVA_PREP.md`. It takes a few
minutes. Every number in the docs comes from these artifacts.

## 5. Run the test suite

From `backend/` (where step 4 left you):

```bash
pytest                      # backend + ML tests
cd ../dashboard && pytest   # dashboard tests (Streamlit AppTest)
cd ../backend               # back to backend/ for the next step
```

The 9 Docker tests are skipped automatically when Docker is not available.

## 6. Set a local API key

Every business API endpoint requires an `X-API-Key` header. `/health` and the automatically generated API documentation remain open.

### Git Bash / macOS / Linux

```bash
export AEGIS_API_KEY=dev-local-key
```

### PowerShell

```powershell
$env:AEGIS_API_KEY="dev-local-key"
```

The value above is only for local development. Use a strong secret in any real deployment.

## 7. Start the FastAPI backend

From `backend/`:

```bash
uvicorn app.main:app --reload --port 8000
```

Open:

- API: `http://127.0.0.1:8000`
- Swagger documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

Example authenticated request:

```bash
curl -H "X-API-Key: $AEGIS_API_KEY" http://127.0.0.1:8000/claims
```

## 8. Start the Streamlit dashboard

Open a **second terminal** at the repository root.

Activate the same virtual environment if necessary, then run:

```bash
cd dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## 9. Local database behavior

When `DATABASE_URL` is not set, Aegis falls back to a local SQLite database:

```text
aegis.db
```

This makes the non-Docker setup suitable for development without requiring PostgreSQL.

To use another database, provide a `DATABASE_URL` environment variable.

---

# Option B — Run With Docker Compose

Docker Compose starts the main application stack together:

- PostgreSQL
- one-shot model/data initialization
- FastAPI backend
- Streamlit dashboard

## 1. Clone the repository

```bash
git clone https://github.com/NinsonTimothy/fraud_detection_for_insurance_claims.git
cd fraud_detection_for_insurance_claims
```

## 2. Create your environment file

### Git Bash / macOS / Linux

```bash
cp .env.example .env
```

### Windows Command Prompt

```cmd
copy .env.example .env
```

### PowerShell

```powershell
Copy-Item .env.example .env
```

Open `.env` and set your own values:

```text
AEGIS_API_KEY=replace-with-a-strong-api-key
POSTGRES_USER=aegis
POSTGRES_PASSWORD=replace-with-a-strong-password
POSTGRES_DB=aegis
```

An optional full `DATABASE_URL` override is also supported.

Do not use the shipped placeholder credentials for any real deployment.

## 3. Build and start the stack

```bash
docker compose up --build
```

The first run may take longer because Docker has to download base images and build the application images.

The `train-init` service prepares the data and model artifacts before the application services start.

Once running:

```text
API:       http://localhost:8000
API docs:  http://localhost:8000/docs
Dashboard: http://localhost:8501
```

The Docker Compose configuration ran successfully before the pre-defence changes; re-run `docker compose up --build` after updating to confirm it on your machine (see `docs/LIMITATIONS.md`).

## 4. Stop the stack

Press:

```text
Ctrl + C
```

Then remove the stopped containers:

```bash
docker compose down
```

To also remove the Docker volumes:

```bash
docker compose down -v
```

> `docker compose down -v` removes persisted Docker volume data. Use it only when you intentionally want a clean reset.

---

## API Authentication

Aegis uses a shared API key for business endpoints.

Send it using:

```text
X-API-Key: <your-key>
```

The following remain publicly accessible for operational convenience:

- `/health`
- FastAPI documentation routes

This is intentionally lightweight authentication for an academic prototype. It is **not** a substitute for production-grade user authentication, RBAC, identity management, or secrets management.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the full security discussion.

---

## Model and Evaluation

### Shipped Model

The champion is chosen by `python -m app.ml.train` using **only the 800-row training split**
(nested, repeated cross-validation, 15 paired folds; each candidate tunes its own
hyperparameters and threshold on inner folds). Candidates: Logistic Regression, Random Forest,
XGBoost, and a one-line baseline rule — *flag if incident severity is "Major Damage"*.
The current champion is recorded in `models/metrics.json` (`primary_model`) and saved as
`models/champion_model.pkl`.

**Key finding:** no ML model beat the one-line rule on F1, and on the test set the shipped model's
review decisions match the rule on every claim. The model's added value is ranking within groups
and per-claim explanations. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

The deployable model excludes the project's identified proxy-style hobby and occupation features by default.

### Current Metric Snapshot

Numbers are deliberately **not** copied here, because they go stale after every retrain. The
authoritative, auto-generated report (including every p-value) is:

[`docs/CURRENT_METRICS.md`](docs/CURRENT_METRICS.md)

Regenerate it from `backend/` with:

```bash
python -m app.ml.train              # several minutes (nested CV; ~20 min on a single core)
python -m app.ml.evaluate_oracle
python -m app.ml.generate_metrics_report
```

### Decision support, not automated decisions

Aegis recommends a level of scrutiny (standard handling / investigator review / priority SIU
review). It never approves, denies, or settles a claim; an investigator makes every decision.

---

## External Validation

A separate external dataset is used to test transfer beyond the model's training distribution.

The model's ranking does not transfer to the external dataset: the internal-test and Oracle ROC-AUC values, their 95% confidence intervals and the verdict (worded automatically from the interval) are in [`docs/CURRENT_METRICS.md`](docs/CURRENT_METRICS.md), section 8. Numbers are deliberately not copied here, so this README cannot go stale after a retrain.

This result is a core finding of the project, not something hidden from the evaluation.

The project analysis attributes much of the failure to **feature availability and distribution shift** between the training data and the external dataset. The external dataset lacks equivalents for several important model features.

Read the detailed interpretation before drawing conclusions:

- [`docs/CURRENT_METRICS.md`](docs/CURRENT_METRICS.md)
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- [`docs/REBUILD_NOTES.md`](docs/REBUILD_NOTES.md)

---

## Explainability

Aegis uses SHAP to provide explanations for model behavior.

The system supports:

- global feature importance
- claim-level explanations
- analyst-facing risk reasons
- model-insight views in the Streamlit dashboard

Explainability should be treated as an aid to analysis rather than proof of causality. A feature receiving a strong SHAP contribution does not establish that the feature causes fraud.

---

## Data

The project includes two main datasets.

### Primary dataset

- 1,000 auto-insurance claims
- approximately 24.7% fraud
- used for model development and internal evaluation

### External validation dataset

- 15,420 claims
- approximately 6% fraud
- used only for external validation/stress testing, not for training the shipped model

The datasets are US auto-insurance data. The project was developed in a University of Ghana academic context, but the current model has **not** been validated on Ghanaian insurance claims.

---

## Important Limitations

Aegis is a research and software-engineering prototype, not a production insurance adjudication system.

Important limitations include:

- Small primary training dataset
- Limited number of confirmed fraud examples
- Significant distribution shift between internal and external datasets
- Poor external generalization
- Dataset-specific feature artifacts
- No demonstrated transferability to Ghanaian insurance claims
- No prior-claims-history features
- No policyholder network-link analysis
- No automated retraining pipeline
- No database migration framework
- Lightweight shared API-key authentication rather than per-user authorization
- Limited PII protection compared with production insurance systems
- No claim-decision governance or human-override framework suitable for real deployment

The project deliberately documents these limitations rather than presenting internal model performance as evidence of production readiness.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the full discussion.

---

## Project History

This repository contains two generations of the project.

### Aegis Risk Engine

The current system is the rebuilt version, with a redesigned:

- ML training and evaluation pipeline
- feature-engineering workflow
- backend API
- database layer
- analyst dashboard
- monitoring workflow
- external-validation process
- test suite
- deployment configuration

### Version A

The earlier implementation is preserved under:

[`legacy/version-a/`](legacy/version-a/)

It remains available for history, comparison, and reference but is not used by the current application.

The original repository state before the rebuild is also preserved in Git history/tagging.

---

## Development Workflow

A typical development cycle is:

```bash
cd backend
pytest
python -m app.ml.run_all     # regenerates models, metrics, reports and docs in order
```

After changing model logic, regenerate the metrics report rather than manually editing reported performance values.

---

## Troubleshooting

### `pg_config executable not found` while installing dependencies

On Windows, this can occur when using Python 3.13 with the pinned `psycopg2-binary==2.9.9`.

Use Python 3.12:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
pip install -r backend/requirements.txt
```

### API starts but business endpoints return an authentication error

Set `AEGIS_API_KEY` and send the same value in the `X-API-Key` header.

### Dashboard does not start

Make sure the dashboard dependencies are installed:

```bash
cd dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Model artifacts are missing

From `backend/`, run:

```bash
python -m app.ml.run_all
```

### Docker services fail on first build

Check that Docker Desktop/Engine is running and that your machine can pull images from Docker Hub, then retry:

```bash
docker compose up --build
```

---

## Documentation

More detailed project documentation is available under [`docs/`](docs/).

Key files include:

- [`docs/CURRENT_METRICS.md`](docs/CURRENT_METRICS.md) — generated model metrics
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — known limitations and deployment caveats
- [`docs/REBUILD_NOTES.md`](docs/REBUILD_NOTES.md) — rebuild decisions and fixes
- [`docs/ml_feature_critique.md`](docs/ml_feature_critique.md) — feature/model critique
- [`docs/generalization_and_cv_results.md`](docs/generalization_and_cv_results.md) — validation analysis

---

## Contributors

### Timothy Ninson

GitHub: [NinsonTimothy](https://github.com/NinsonTimothy)

### Kwabena Adipah Osei

GitHub: [adipahosei](https://github.com/adipahosei)

---

## Responsible Use

Fraud-risk scores should support human investigation, not replace it.

A high score is not proof that a claim is fraudulent, and a low score is not proof that a claim is legitimate. Any real-world deployment would require additional validation, governance, privacy safeguards, fairness review, access controls, operational monitoring, and human oversight.
