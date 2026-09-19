# Known, disclosed limitations

This file is deliberately part of the delivered project, not an afterthought —
a DCIT400 defense goes better when limitations are stated precisely and
proactively than when a panel member finds them first.

## Data & modeling

- **Small training set.** 1,000 rows, ~247 fraud cases, 74 engineered
  features after an earlier 122-feature version was found to overfit (see
  `docs/ml_feature_critique.md` and the Model Insights page's audit tab).
  Every metric in this project carries real uncertainty at this sample
  size — report the 5-fold CV range, not just the point estimate.
- **`zip3_risk_tier` is documented, disclosed target-encoding leakage,
  not fixed.** It's built from the same 1,000 rows the model trains on and
  is the single largest share of the model's SHAP weight (see Model
  Insights). The honest fix (recompute per-CV-fold, or drop it) is a
  one-line change in `backend/app/ml/train.py` — left as an explicit,
  disclosed design choice rather than silently applied, so the "what would
  actually change if you fixed this" question has a real answer.
- **`is_highrisk_hobby` / `is_exec_occupation` are dataset artifacts, not
  demonstrated fraud signal** — see Model Insights → Feature quality audit.
  Kept because dropping them silently would misrepresent what this build's
  shipped model actually does; flagged individually so they can be removed
  in one line.
- **`incident_severity`'s ordinal encoding is inferred, not confirmed**
  against an original source labeling scheme (marked `TODO-VERIFY` in
  `feature_engineering.py`).
- **External validation (Oracle) shows the model does not generalize past
  its own training distribution** — ROC-AUC collapses from ~0.66 (internal)
  to ~0.48 (Oracle, statistically random). Root cause is quantified: ~98%
  of the model's SHAP weight sits on features that are constant once
  Oracle-mapped data passes through, because Oracle has no ZIP code, no
  incident-severity field, and no claim-dollar breakdown. A stress test
  (fresh models trained directly on Oracle's own fields, ROC-AUC ~0.81)
  confirms this is a feature-availability problem, not evidence Oracle
  itself is unlearnable. See the Monitoring page for the full breakdown.
- **No prior-claims-history / fault-attribution / network-link features** —
  the primary dataset has no policyholder ID linking multiple claims, so
  "how many claims has this person filed before" (the single most-cited
  real fraud red flag per NICB) cannot be computed from it at all. This is
  a data-collection ceiling, not a missing line of code.

## Engineering

- **No authentication/authorization on the API.** Anyone who can reach
  `:8000` can score claims, read the audit log, and export investigator
  feedback. Fine for a thesis demo; would need real auth for any actual
  deployment.
- **Dynamic cost threshold uses a flat, editable analyst-review-time proxy**
  (`FP_REVIEW_COST` in `backend/app/core/config.py`, default 250 currency
  units) — a disclosed modeling assumption, not independently cited, same
  pattern as the sibling MoMo Guard project.
- **Retraining loop is manual**, not automatic: export the investigator
  feedback CSV (`GET /feedback/export`), append confirmed labels to the
  training set, re-run `python -m app.ml.train`.
- **No database migrations.** `db/session.py::init_db()` calls SQLAlchemy's
  `create_all()`, which creates missing tables but never alters an existing
  one — a schema change after first run needs a manual DB reset, not an
  upgrade path. Acceptable for a thesis-scale single-environment SQLite
  deployment; a real multi-environment deployment would need Alembic (or
  equivalent) migrations instead.
- **Kafka consumer is at-least-once, not exactly-once** — a handler failure
  part-way through a consumed batch leaves already-processed messages ahead
  of it processed; the batch is not retried from the start.
- **Minimal PII handling** — claim payloads are stored as-is in the
  `claims.raw_payload` JSON column, no field-level encryption or masking.

## Docker / this build's sandbox specifically

`dockerd` **can** start and run in the sandbox this project was assembled
in — but Docker Hub (`registry-1.docker.io`) is network-blocked by that
sandbox's egress policy, so no base image (Python, Postgres, Kafka — all of
them) can actually be pulled or built there. `docker compose -f
docker-compose.yml config` DOES validate the compose file's full syntax and
service graph successfully (verified during this build). On a machine with
normal Docker Hub access, `docker compose up --build` should work
end-to-end without further changes — this was not independently verified
end-to-end outside the sandbox, so budget time to debug on first real run
the way any un-execute-tested deployment config deserves.
