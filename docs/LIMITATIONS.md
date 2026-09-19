# Known, disclosed limitations

This file is deliberately part of the delivered project, not an afterthought —
a DCIT400 defense goes better when limitations are stated precisely and
proactively than when a panel member finds them first.

## Data & modeling

- **Small training set.** 1,000 rows, ~247 fraud cases; the shipped model's
  exact feature count is in `models/metrics.json`'s `n_features` (70 with
  the deployable default of proxy features excluded, see the bullet
  below — was 74 when they were included, after an earlier 122-feature
  version was found to overfit — see `docs/ml_feature_critique.md` and
  the Model Insights page's audit tab). Every metric in this project
  carries real uncertainty at this sample size — report the 5-fold CV
  range, not just the point estimate.
- ~~`zip3_risk_tier` is documented, disclosed target-encoding leakage, not
  fixed~~ — **resolved (PB-02):** it was worse than disclosed leakage. This
  dataset's `insured_zip` values are 6-digit US ZIPs, not the 5-digit form
  the original `// 100` prefix logic assumed, so the resulting "ZIP3" was
  actually a near-row-unique 4-digit prefix (515 groups from 1,000 rows,
  median group size 2.0). It let the model memorize labels on train
  (0.2%-71.8% fraud rate by tier) while collapsing to a flat ~20-28% on
  test, and it had absorbed 53.2% of total SHAP weight. There is no honest
  version of a near-row-unique lookup table to keep and disclose, so it has
  been removed entirely rather than recomputed per-fold. See
  `docs/REBUILD_NOTES.md` for the full reproduction evidence and
  before/after metrics.
- **`is_highrisk_hobby` / `is_exec_occupation` are dataset artifacts, not
  demonstrated fraud signal** — see Model Insights → Feature quality audit.
  **Resolved (SH-02 / D3):** these are no longer baked into the shipped
  model at all. `feature_engineering.engineer_features()` gates both
  behind `INCLUDE_PROXY_FEATURES` (`app/core/config.py`, default
  **`False`**) — the deployable headline model genuinely never computes or
  sees them, not merely a flag that could be removed in one line.
  Including them is opt-in (`INCLUDE_PROXY_FEATURES=true`), e.g. to
  reproduce the comparison below. This has a real, measured cost: RF
  holdout ROC-AUC drops from ~0.85 (proxy features on) to ~0.79 (proxy
  features off), and recall/F1 drop similarly — see
  `data/processed/proxy_feature_ablation.csv` and
  `docs/REBUILD_NOTES.md` §"SH-02" for the full before/after numbers, both
  the single holdout split and 5-fold CV. This project's own judgment is
  that shipping a model that leans on an unexplained lifestyle/occupation
  → risk association — close to proxy-discrimination, the kind real
  insurance regulators scrutinize insurers for — is not worth that
  performance gain, so the honest, disclosed trade-off is accepted rather
  than hidden.
- ~~`incident_severity`'s ordinal encoding is inferred, not confirmed~~ —
  **resolved (PB-24):** confirmed directly against the original FYP
  project's `02_preprocessing.ipynb`/`03_modelling.ipynb` source (Trivial
  Damage < Minor Damage < Major Damage < Total Loss, via
  `sklearn.OrdinalEncoder`). `feature_engineering.py`'s `SEVERITY_ORDINAL`
  matches it exactly; the `TODO-VERIFY` comment has been removed.
- **External validation (Oracle) shows the model does not generalize past
  its own training distribution.** Root cause is feature-availability, not
  a data problem: Oracle has no incident-severity field and no claim-dollar
  breakdown, so every feature derived from those collapses to a constant
  fallback on Oracle-mapped data. A stress test (fresh models trained
  directly on Oracle's own fields) confirms Oracle itself is learnable
  fraud data. **PB-02 note:** an earlier version of this bullet also
  attributed a large share of this collapse to `zip3_risk_tier` (no ZIP
  code in Oracle) — that feature has since been removed entirely as a
  near-row-unique leakage bug, not kept as disclosed leakage, so it's no
  longer a contributing factor. Exact ROC-AUC/SHAP-share figures are
  regenerated from `models/metrics.json` and the Oracle validation report
  (see the Monitoring page), not hand-typed here.
- **No prior-claims-history / fault-attribution / network-link features** —
  the primary dataset has no policyholder ID linking multiple claims, so
  "how many claims has this person filed before" (the single most-cited
  real fraud red flag per NICB) cannot be computed from it at all. This is
  a data-collection ceiling, not a missing line of code.

## Engineering

- ~~No authentication/authorization on the API. Anyone who can reach
  :8000 can score claims, read the audit log, and export investigator
  feedback.~~ — **resolved (PB-11):** every business endpoint (scoring,
  claims, feedback, audit, monitoring — `/health` and the auto-generated
  docs routes stay open) now requires a matching `X-API-Key` header
  (`app/core/security.py`, `app/core/config.py`'s `AEGIS_API_KEY`,
  `docker-compose.yml`'s `api` service). This is a single, deployment-wide
  shared secret, not per-user auth/RBAC — genuinely still out of scope for
  a thesis-scale project, and disclosed as such, but "reachable with zero
  credential at all" is closed. The shipped default
  (`CHANGE-ME-insecure-default-api-key`) is deliberately an obvious
  placeholder so "nobody set a real key" is a visible fact about a
  deployment, not a silently-working default that looks secure.
- **Dynamic cost threshold uses a flat, editable analyst-review-time proxy**
  (`FP_REVIEW_COST` in `backend/app/core/config.py`, default 250 currency
  units) — a disclosed modeling assumption, not independently cited, same
  pattern as the sibling MoMo Guard project. **PB-03:** under that
  assumption, the cost-optimal threshold search reliably lands at the very
  bottom of the swept range ("flag nearly everyone") because this
  dataset's mean claim amount is ~211x the flat review cost — a
  reproduced, expected property of a pure expected-cost objective at this
  cost ratio, not a bug. It's reported as a diagnostic/sensitivity number
  in `models/metrics.json`, not used as the operating threshold (which is
  chosen separately, by F1-optimal search on honest out-of-fold training
  predictions — see `docs/REBUILD_NOTES.md` §8).
- **Retraining loop is manual**, not automatic: export the investigator
  feedback CSV (`GET /feedback/export`), append confirmed labels to the
  training set, re-run `python -m app.ml.train`.
- **No database migrations.** `db/session.py::init_db()` calls SQLAlchemy's
  `create_all()`, which creates missing tables but never alters an existing
  one — a schema change after first run needs a manual DB reset, not an
  upgrade path. Acceptable for a thesis-scale single-environment SQLite
  deployment; a real multi-environment deployment would need Alembic (or
  equivalent) migrations instead.
- ~~Kafka consumer is at-least-once, not exactly-once~~ — **resolved
  (PB-08/PB-09, D1):** the entire simulated-Kafka-stream ingestion path
  (`app/api/ingestion.py`, `app/kafka/`) has been removed, not fixed —
  no live broker was ever reachable in this build's sandbox to verify the
  producer/consumer wiring against, so "logic verified directly, wiring
  unverified" understated the risk of shipping an entire
  never-integration-tested subsystem. Claims now enter via `/score`
  (single JSON) and `/score/batch` (CSV upload) only — both fully tested
  against a live `TestClient`. See `docs/REBUILD_NOTES.md`.
- **Minimal PII handling** — claim payloads are stored as-is (unencrypted)
  in the `claims.raw_payload` JSON column; no field-level encryption
  at rest. **Partially resolved (PB-11):** `insured_zip` (the one
  quasi-identifier in this schema — no name/SSN/DOB field exists at all)
  is now masked in every API response that echoes a claim's raw payload
  back (`GET /claims`, `GET /claims/{id}` — `app/core/security.py`'s
  `mask_pii()`), applied at the response boundary only; what's stored and
  what scoring reads are both untouched. `GET /feedback/export`
  deliberately does NOT mask it — that endpoint's whole purpose is
  producing a file meant to be appended straight into the real training
  data (though no feature currently reads `insured_zip` at all, see
  PB-02) — but it's no longer reachable without the same API key as
  everything else. Encryption-at-rest and masking for any future
  PII-bearing field remain out of scope.

## Docker / this build's sandbox specifically

`dockerd` **can** start and run in the sandbox this project was assembled
in — but Docker Hub (`registry-1.docker.io`) is network-blocked by that
sandbox's egress policy, so no base image (Python, Postgres — the compose
file's only two pulled images since PB-08/PB-09 removed the Kafka
service) can actually be pulled or built there. `docker compose -f
docker-compose.yml config` DOES validate the compose file's full syntax and
service graph successfully (verified during this build). On a machine with
normal Docker Hub access, `docker compose up --build` should work
end-to-end without further changes — this was not independently verified
end-to-end outside the sandbox, so budget time to debug on first real run
the way any un-execute-tested deployment config deserves.
