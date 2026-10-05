# Known, disclosed limitations

This file is deliberately part of the delivered project, not an afterthought —
a DCIT400 defense goes better when limitations are stated precisely and
proactively than when a panel member finds them first.

## Pre-defence findings (read these first)

- **The model does not beat a one-line rule at deciding which claims to
  review.** Leak-free selection (training split only, nested CV, 15 paired
  folds) included the rule *flag if incident_severity = "Major Damage"* as a
  baseline. No ML candidate beat it on F1 (Random Forest −0.004, corrected
  p = 0.21; exact figures in `docs/CURRENT_METRICS.md`). On the 200 test
  claims the shipped model's review/no-review decision is identical to the
  rule's on every claim. Reason: in this dataset Major Damage claims are
  ~60% fraud and every other severity is 7–13%, so one field carries almost
  all the decision-level signal. What the model adds is (a) a ranking
  *within* groups — the rule gives every Major Damage claim the same score,
  so it cannot say which of ~80 to open first; PR-AUC is higher by +0.035,
  though not significantly — and (b) a per-claim explanation. The system
  is therefore presented as decision support, not as a better decider.
- **The witness pattern runs the "wrong" way, and we treat it as a dataset
  artefact.** Fraud rate by number of witnesses: 0 → 20.1%, 1 → 24.4%,
  2 → 29.6%, 3 → 24.7% (≈250 claims each). Real-world guidance treats *no*
  witnesses as the red flag. The engineered `is_no_witness` flag encoded that
  real-world assumption and so pushed scores in the direction opposite to its
  name; it has been removed. The raw `witnesses` count stays a feature, so
  the model can still use whatever the data shows, but any reason code saying
  "more witnesses raised the risk" reflects this dataset, not fraud
  behaviour, and should not be generalised. Supporting evidence that it is an
  artefact: on the independent Oracle dataset the relationship reverses
  (a witness present: 3.4% fraud vs 6.0% without).
- **On Oracle, the ranking is significantly inverted, not random.** ROC-AUC's
  bootstrap 95% CI lies entirely below 0.5 (current values in
  `docs/CURRENT_METRICS.md`; the wording there is derived from the CI by
  code). In plain terms: on that dataset the model tends to give genuine
  fraud cases slightly *lower* scores than legitimate ones. The most
  plausible reason is that almost all of its weight sits on fields Oracle
  lacks, and among the few shared fields the witness relationship is
  reversed (above). Earlier documents that called this "random" have been
  annotated.
- **The test set is now touched once.** Before this fix, the champion
  comparison cross-validated on all 1,000 rows, so test rows influenced
  which model shipped. Selection now uses the 800 training rows only.
- **Decision support only.** No risk band leads to automatic approval or
  denial. The Low band recommends normal claims handling; a person decides.

## Data & modeling

- **Small training set.** 1,000 rows, ~247 fraud cases; the shipped model's
  exact feature count is in `models/metrics.json`'s `n_features` (69 after the pre-defence removal of `is_no_witness`; 70 before it, with
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
  (see the Monitoring page), not hand-typed here — run
  `python -m app.ml.generate_metrics_report` (PB-15) from `backend/`, or
  read its output at `docs/CURRENT_METRICS.md`, for the current numbers
  rather than trusting any figure typed directly into this file.
- **Statistical caveat on the Oracle comparison (SH-05).** Oracle's 15,420
  rows give the ROC-AUC comparison itself high statistical power — the
  bootstrap 95% CI on Oracle ROC-AUC (`docs/CURRENT_METRICS.md`) is tight
  enough to say with confidence whether it contains 0.5 or not, which is
  the headline claim ("does not generalize"). But **PR-AUC is not
  comparable across the two datasets the same way** — average precision's
  own baseline shifts with class prevalence, and this project's internal
  fraud rate (~24.7%) is far higher than Oracle's (~6.0%), so part of any
  internal-vs-Oracle PR-AUC gap reflects that prevalence difference, not
  model degradation alone. Read the ROC-AUC comparison and the SHAP
  constant-feature-share analysis (both prevalence-independent) as the
  primary evidence for the generalization failure; treat the PR-AUC gap
  as directionally consistent with, but not an independently quantified
  confirmation of, that same finding.
- **Geographic / regulatory transferability (PB-23).** Both training
  datasets (the primary 1,000-row set and the Oracle external-validation
  set) are US auto-insurance claims — US dollar amounts, US state codes
  (`policy_state`/`incident_state`, e.g. OH/IN/IL), and US-specific
  categorical fields (`police_report_available`, `authorities_contacted`
  in terms recognizable to a US claims process). This project is produced
  in a University of Ghana academic context, but no Ghanaian claims data,
  currency, or regulatory framework was used, mapped, or validated against
  anywhere in this pipeline — unlike the sibling MoMo Guard project, which
  is explicitly grounded in Ghanaian mobile-money data and Bank of Ghana
  statistics. Nothing in this repo's methodology, feature set, or reported
  numbers should be read as evidence that the model — or even the general
  approach — transfers to the Ghanaian insurance market; that would need
  its own dataset and its own external-validation exercise, exactly like
  the Oracle adapter provides for the (still US-only) generalization
  question this project does answer.
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

**Fixed in this pass (PB-13, D2 — SQLite default, Postgres optional):**
`postgres`'s `5432` used to be published to the host (`"5432:5432"`) for
no reason this topology needs — every consumer reaches it over the
compose-internal network by service name — while carrying the default
`aegis`/`aegis` credentials; the mapping is gone, so a `docker compose up`
no longer exposes a trivially-guessable-credential DB to the host
network by default. Those credentials were also hardcoded in four
separate places (the `postgres` service plus `api`/`dashboard`/
`train-init`'s `DATABASE_URL`s) with no override mechanism; all four now
read `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` (or a full
`DATABASE_URL` override) from the shell/`.env.example` so they can't
drift apart and can be changed in one place. `train-init` (runs
`clean_data.py`/`train.py`/`evaluate_oracle.py` — none of which import
anything DB-related, verified by inspection) no longer waits on
Postgres's healthcheck or carries an unused `DATABASE_URL`, so training
isn't blocked by, or made to fail alongside, a database it never
touches. None of this changes `config.py`'s own SQLite-by-default
behavior — it was already true, just not documented as the deliberate
D2 decision it is; see `docs/REBUILD_NOTES.md`.
