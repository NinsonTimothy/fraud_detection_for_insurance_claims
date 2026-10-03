# Changelog — full bug-fix/hardening pass

This file is the final deliverable summary for the PB-xx/SH-xx bug-fix and
hardening pass run against the baseline commit (`c6543b8`, "chore: baseline
commit of Aegis Risk Engine, pre-fix"). Three things, in order: (1) an
honest before/after metrics comparison, generated from real artifacts on
both ends, not hand-typed; (2) every ticket ID mapped to the exact commit
that closed it; (3) an incomplete-ticket report — the backlog items that
were NOT implemented, and why, stated as plainly as everything else in this
repo's `docs/`.

## 1. Before / after

**Before** = `models/metrics.json` / `data/external/oracle/oracle_validation_report.json`
/ `data/processed/cross_validation_results.csv` as committed at `c6543b8`
(the pre-fix baseline). **After** = the same artifacts as they stand today —
regenerate with `python -m app.ml.generate_metrics_report` (PB-15) from
`backend/`, or read `docs/CURRENT_METRICS.md`, rather than trusting the
numbers below to still be current by the time this is read.

### Held-out single-split (Random Forest, shipped model)

| Metric | Before (`c6543b8`) | After (current) | Why it moved |
|---|---|---|---|
| Operating threshold | 0.48 | 0.44 | PB-03: threshold now chosen from honest out-of-fold training predictions, never the test set |
| Recall | 49.0% | 73.5% | PB-02 (leakage removal) + PB-03 (honest threshold) + SH-01 (`authorities_contacted` NaN-misparsing fix) + PB-24 (`incident_severity`/`insured_education_level` encoding fixes) combined — see each ticket's own before/after in §2 |
| Precision | 54.5% | 63.2% | — |
| F1 | 0.516 | 0.679 | — |
| PR-AUC | 0.450 | 0.545 | — |
| ROC-AUC | 0.656 | 0.794 | — |
| Accuracy | 77.5% | 83.0% | — |
| n_features | 74 | 70 | PB-02 removed `zip3_risk_tier` (leakage); SH-02/D3 gates `is_highrisk_hobby`/`is_exec_occupation` off by default |

### 5-fold cross-validation, full pipeline refit per fold (Random Forest)

| Metric | Before | After |
|---|---|---|
| ROC-AUC | 0.666 ± 0.042 | 0.770 ± 0.021 |
| PR-AUC | 0.437 ± 0.047 | 0.514 ± 0.045 |
| F1 | 0.483 ± 0.064 | 0.639 ± 0.047 |
| Recall | 0.421 ± 0.082 | 0.676 ± 0.054 |
| Precision | 0.586 ± 0.065 | 0.608 ± 0.055 |

Read this pair the way §3 of `docs/REBUILD_NOTES.md` already frames it:
the "before" CV numbers were themselves measured under a since-fixed CV
leakage bug (`zip3_lookup` built once on the full pool and reused across
folds) for an even earlier snapshot — the `c6543b8` baseline above already
reflects that specific fix (its CV std/mean are in the honest, single-holdout-
consistent range, not the ~0.94 leaked figure `docs/REBUILD_NOTES.md` §2
documents from an even earlier state). Every number in the "after" column
moved up, not because thresholds were cherry-picked to look better, but
because three real, reproduced bugs were fixed along the way (`zip3`
leakage, `authorities_contacted` "None"-as-NaN misparsing, and — the
largest single contributor — `incident_severity`'s and
`insured_education_level`'s encoding fixes, PB-24).

### Oracle external validation

| Metric | Before | After |
|---|---|---|
| Internal holdout ROC-AUC | 0.656 | 0.794 |
| Oracle ROC-AUC | 0.483 | 0.463 |
| Oracle ROC-AUC 95% bootstrap CI | not computed at baseline | [0.443, 0.481] (SH-04) |
| Oracle PR-AUC | 0.058 | 0.055 |
| Features constant on Oracle-mapped data | 61 of 74 (share of SHAP weight: 97.6%) | 58 of 70 (share of SHAP weight: 91.7%) — regenerate via `python -m app.ml.evaluate_oracle` from `backend/`; this breakdown lives in `data/external/oracle/oracle_validation_report.json`, not duplicated by `generate_metrics_report.py` |
| Oracle rows / fraud rate | 15,420 / 6.0% | unchanged (same fixed external dataset both times) |

**Read honestly:** the Oracle ROC-AUC did not improve alongside the
internal metrics — if anything it's marginally lower, and (PB-15/SH-05)
the CURRENT bootstrap CI no longer even contains 0.5, meaning the
generalization gap is now measurably *below* chance rather than merely
statistically indistinguishable from it. This is expected, not a
regression this pass caused: every fix in this backlog improved the
internal model's use of fields Oracle has never had (`incident_severity`,
`insured_education_level`, ZIP-derived features) — fixing how well the
model uses those fields was never going to help it generalize to a
dataset that doesn't supply them. See `docs/LIMITATIONS.md`'s Oracle
bullets and PB-23's geographic-transferability disclosure for the full,
disclosed picture.

### Test coverage

| | Before | After |
|---|---|---|
| Backend tests | 16 (`test_api.py`, `test_kafka_logic.py`, `test_ml_core.py`) | 137, across 15 files |
| Dashboard tests | 0 (no `dashboard/tests/` at all) | 11 |
| Kafka ingestion path | present (simulated, never integration-tested against a live broker) | removed entirely (D1/PB-08/PB-09), not merely deprecated |

## 2. Ticket → commit map

Every commit message already carries its own detailed before/after
evidence in its body and in the matching `docs/REBUILD_NOTES.md` section —
this table is the index, not a replacement for reading either. Run
`git log <hash> -1` for any row to see the full message.

| Ticket | Commit | Summary |
|---|---|---|
| Baseline | `c6543b8` | Pre-fix baseline commit (Phase 0) |
| PB-17 | `5db8ba4` | Remove dead config/artifacts, wire up unused settings |
| PB-21 | `6772b87` | Doc/code drift and copy-paste leftovers from sibling project |
| PB-22 | `54b872b` | Add `.gitignore`/CI, timezone-aware timestamps, migrations note |
| PB-24 | `3c95e57` | Confirm `incident_severity` ordinal order, remove TODO-VERIFY; fix `insured_education_level` encoding |
| PB-25 | *(no repo commit — see note below)* | Purge stale figures from the `aegis_risk_engine_build_summary.md` project doc |
| PB-01 | `2a71a65` | Fix dashboard `sys.path` crash on first scoring action |
| PB-02 | `24e88a1` | Remove `zip3_risk_tier` 4-digit-prefix leakage feature entirely, retrain |
| SH-01 | `1f7283e` | `authorities_contacted`'s "None" is a real category, not NaN |
| PB-03 | `c4879f6` | Stop choosing thresholds/SHAP importance from the test set |
| PB-14, SH-03, SH-06 | `63b8961` | Evidence-backed champion selection (D4), SMOTE vs. class-weight comparison, fix `is_new_customer` threshold |
| PB-04 | `e54f8a4` | Unify risk policy — `score_batch`/`score_one` used to disagree at risk-band edges |
| PB-06 | `4cf11ec` | Validate `/score` input instead of accepting `dict[str, Any]` |
| PB-07 | `e2d6250` | Expected client mistakes (duplicate ref, empty CSV, non-numeric cell) 500'd instead of returning a clean 4xx |
| PB-10 | `191150f` | Fix invalid PSI comparison (scaled vs. unscaled features) + masked bool-dtype crash |
| PB-05 | `bcca5c8` | Restore/upgrade local SHAP explanations (k=8, not 3) |
| SH-02 | `354e402` | Gate `is_highrisk_hobby`/`is_exec_occupation` behind `INCLUDE_PROXY_FEATURES` (D3), off by default |
| SH-04 | `35c56e4` | Report uncertainty (5-fold mean±SD, bootstrap 95% CI) everywhere a point estimate is reported |
| PB-08, PB-09 | `8d1f060` | Remove the simulated-Kafka ingestion path entirely (D1) |
| PB-11 | `3e59f59` | Add API-key auth; mask PII (`insured_zip`) in claim responses |
| PB-12 | `9b38c0e` | Dashboard scores and escalations now persist to the DB (were silently discarded) |
| PB-13 | `94941da` | Fix Docker/deployment config — credentials, exposed DB port, unnecessary train-init/Postgres coupling (D2) |
| PB-18 | `a4f5ce2` | Batch scoring: N+1 inserts, missing `top_reasons`, unbounded upload size/row count |
| PB-19 | `025674e` | Dashboard hardcoded scoring dates; free-text hobby/occupation fields |
| PB-20 | `12d1992` | Train/serve skew from per-batch median fallbacks; unsafe batch zip/date parsing |
| PB-16 | `1b52283` | Oracle adapter, train-pipeline, train/serve-parity, and negative-path test coverage |
| PB-15, SH-05, PB-23 | `a109741` | Auto-generated metrics doc; Oracle statistical (PR-AUC prevalence) caveat; geographic/regulatory-transferability disclosure |

**PB-25 note:** this ticket's deliverable was an edit to the
`aegis_risk_engine_build_summary.md` document held in the claude.ai FYP
project (adding the "STALE — SUPERSEDED" banner correcting that document's
now-outdated figures) — that document lives outside this git repository,
so there is no corresponding commit here. Its content is not duplicated
into this repo; `docs/REBUILD_NOTES.md` and `docs/CURRENT_METRICS.md` are
this repo's own, always-current sources of truth.

**Decisions (D1–D4)**, applied via the tickets above, not separate rows:
D1 (remove Kafka, not just leave it unwired) → PB-08/PB-09 (`8d1f060`);
D2 (SQLite default, Postgres optional) → PB-13 (`94941da`); D3 (proxy
features behind a flag, off by default) → SH-02 (`354e402`); D4 (champion
model chosen by evidence, not assumed) → PB-14 (`63b8961`).

## 3. Incomplete-ticket report

**Every ticket in the PB-01–PB-25 / SH-01–SH-06 backlog was implemented**
(31 of 31 — see the map above; PB-25's deliverable lives outside this repo
per the note above, but it was completed). Nothing was silently dropped or
left half-done. That said, several tickets' own fixes disclosed further
work that was deliberately **not** attempted, stated here rather than left
implicit, matching this project's own "name the ceiling, don't quietly
work around it" standard:

- **Prior-claims-history / fault-attribution / network-link features**
  (named in `docs/insurance_fyp_ml_feature_critique.md` and
  `docs/LIMITATIONS.md`) were never added — the primary 1,000-row dataset
  has no policyholder ID linking multiple claims, so this isn't a
  remaining line of code, it's a data-collection ceiling this backlog
  can't close.
- **No cross-dataset retrain** (primary + Oracle, on their ~10/35-field
  schema intersection) was attempted, per
  `docs/insurance_fyp_project_handoff.md`'s own recommendation to scope
  that as a separate, deliberate experiment rather than assume it's "the
  fix" — real risk of trading internal accuracy for external validity,
  not evaluated here.
- **No feature-dropout / simulated-missingness augmentation** during
  training was implemented (the other concrete generalization-improvement
  idea named in the same handoff doc) — out of scope for this bug-fix
  pass, which fixed correctness/honesty issues in the existing pipeline
  rather than attempting a new training methodology.
- **Docker Compose was not independently verified end-to-end** — `docker
  compose config` validates the full compose file (done, PB-13), but no
  base image could actually be pulled in this build's sandbox (Docker Hub
  network-blocked); see `docs/LIMITATIONS.md`'s Docker section.
- **No database migrations (Alembic or equivalent).** `db/session.py`'s
  `create_all()` creates missing tables but never alters an existing one —
  disclosed as acceptable for a thesis-scale single-environment SQLite
  deployment, not fixed, per `docs/LIMITATIONS.md`.
- **No per-user auth/RBAC.** PB-11 closed the "reachable with zero
  credential at all" gap with a single shared `AEGIS_API_KEY`, explicitly
  not a full identity system — disclosed, not attempted further.

None of the above were backlog tickets that got skipped; they're
out-of-scope items each ticket's own fix surfaced and explicitly declined
to also solve, consistent with how `docs/LIMITATIONS.md` has disclosed
them throughout this pass.
