# Pre-defence fixes (October 2026)

One entry per item on the supervisor-feedback to-do list. Numbers are NOT
copied here (they go stale); read `docs/CURRENT_METRICS.md` and
`docs/THESIS_NUMBERS.md`, both generated from the artifacts.

| ID | To-do item | What changed | Where | Locked in by |
|---|---|---|---|---|
| VZ-01 | Visualise the scoring result | Risk gauge with review threshold; contribution chart (one bar per claim field, red raises / green lowers, labelled with the claim's value); "this claim vs. 200 test claims" histogram with percentile; claim-amount donut; plain-English "pushing up / pulling down" summary | `dashboard/app_pages/score_claim.py`, `dashboard/components/charts.py` | `dashboard/tests` |
| MS-02 | Witness pattern | `is_no_witness` removed (it assumed "no witnesses = suspicious"; here 0 witnesses has the LOWEST fraud rate). Raw `witnesses` kept. Pattern documented as a dataset artefact | `feature_engineering.py`, `docs/LIMITATIONS.md` | `test_predefence_fixes.py::test_is_no_witness_is_gone_but_witness_count_remains` |
| EX-01 | Police-report label bug | One-hot dummies are grouped back to their source field (SHAP is additive, so contributions are summed) and labelled with the claim's ACTUAL category. "NO" can no longer surface as "YES (0)" | `backend/app/ml/explainer.py` | `test_police_report_reason_shows_the_answer_actually_given`, `test_one_hot_reasons_are_grouped_and_additive` |
| TC-01 | Auto total | `total = injury + property + vehicle` (true for all 1,000 training rows). Derived in the shared pipeline, shown live in the form, API rejects a contradicting total (422) | `feature_engineering.derive_total_claim_amount`, `schemas.py`, `score_claim.py` | 3 tests in `test_predefence_fixes.py`, `test_score_claim_total_is_calculated_not_typed` |
| SC-01 | Scoring form fields | Form now asks for every field the model uses (incident state, umbrella limit, bodily injuries, collision type, authorities, policy state/CSL, education, relationship, capital gains/loss, hour, vehicles, property damage...). Hobby/occupation/ZIP removed (unused). Category options come from the model's own columns. Page shows coverage of model weight | `score_claim.py` | `test_score_claim_form_asks_for_model_fields_not_unused_ones` |
| BR-01/02 | Batch review dashboard | KPI cards, 6 charts, sidebar filters, top-20 table with reasons + drill-down, escalation, 4 downloads, built-in sample batch. Bug fixed: the page used to re-score and re-save the whole file on every widget change | `batch_review.py`, `charts.py` | `test_batch_review_*` (4 tests) |
| MS-01 | Fair, leak-free model selection + rule baseline | Selection on the 800 training rows only; nested + repeated CV (15 paired folds); each candidate tunes its own hyperparameters and threshold; one-line Major-Damage rule included; pre-declared rule; Nadeau-Bengio corrected t-test; test set used once | `model_selection.py`, `train.py` | `test_selection_artifacts_never_used_the_test_split`, `test_rule_baseline_flags_exactly_major_damage` |
| MS-01 | Stale p-values | Old comparison + hardcoded "p=0.0086" note removed; every p-value written by code to `champion_pairwise_tests.csv` | `model_selection_experiments.py`, `generate_metrics_report.py` | `test_report_p_values_come_from_the_computed_tests_csv_not_typed_text` |
| OR-01 | Oracle wording | Verdict derived from the bootstrap CI: entirely below 0.5 → "significantly inverted". Applied in the report, Overview and Monitoring pages; historical docs annotated | `evaluate_oracle.py`, `data_access.oracle_roc_verdict`, pages | `test_report_flags_when_oracle_ci_excludes_random_chance` |
| DS-01 | Decision-support wording | "auto-approved" removed; every action is a recommendation; notice returned with every score and shown on both scoring pages | `risk_policy.py`, `inference.py`, `schemas.py` | `test_no_action_text_claims_automatic_approval`, `test_score_one_returns_decision_support_notice` |
| TN-01 | Thesis numbers | `python -m app.ml.generate_thesis_numbers` writes an old → new table for every figure the thesis quotes | `generate_thesis_numbers.py`, `docs/THESIS_NUMBERS.md` | — |

## Outcome of the honest selection

No ML candidate beat the Major-Damage rule on F1, and the shipped Random
Forest makes the same review decision as the rule on every test claim. Per
the pre-declared rule, the best ML candidate (Random Forest) still ships,
and the finding is disclosed in `docs/LIMITATIONS.md`, on the Overview page
and on Model insights → Model selection. The model's contribution is
ranking within groups and per-claim explanation.

## Also changed

- The shipped model is loaded from `models/champion_model.pkl`; inference,
  Oracle evaluation and every dashboard page follow `metrics.json`'s
  `primary_model` rather than assuming Random Forest.
- `use_container_width` (deprecated in Streamlit 1.62) replaced with `width="stretch"`.
- Model step names unified to `clf` inside every pipeline.
