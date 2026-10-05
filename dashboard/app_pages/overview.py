"""app_pages/overview.py — real, reproducible model-performance KPIs, no
fabricated production numbers. Every figure here is either a real
evaluation metric from models/metrics.json or a real session count."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    # PB-01: append, never insert(0, ...) — inserting the dashboard
    # dir at the FRONT of sys.path on every rerun is what let
    # `import app...` resolve to the old dashboard/app.py instead of
    # the backend's app/ package (see components/data_access.py,
    # which puts backend/ at sys.path[0] once, on first import).
    sys.path.append(_dashboard_root)
from components.data_access import bootstrap_ci_for, load_bootstrap_ci, load_champion_decision, load_cross_validation, load_metrics, load_model_comparison, models_are_available, oracle_results_available, oracle_roc_verdict, load_oracle_report
from components.theme import inject_css, kpi_card, page_header, MUTED, DANGER, WARNING

inject_css()
page_header("Overview", "Real, reproducible metrics from the last training run — nothing here is a fabricated production number.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

metrics = load_metrics()
comparison = load_model_comparison()
# MS-01: the shipped model is whichever candidate won leak-free selection.
CHAMPION = metrics.get("primary_model", "random_forest")
rf_row = comparison[comparison["model"] == CHAMPION].iloc[0]

# SH-04: every headline number below is a single point estimate off ONE
# fixed 200-row holdout split — attach a bootstrap 95% CI (resampling the
# same split) so the KPI cards don't imply more precision than a 200-row
# split can actually support. cv_row's *_std gives the complementary
# across-FOLD view (see backend/app/ml/uncertainty.py's module docstring
# for why these are two different, both-worth-reporting questions).
bootstrap_df = load_bootstrap_ci()
cv_df = load_cross_validation()
cv_row = cv_df[cv_df["model"] == CHAMPION]
cv_row = cv_row.iloc[0] if not cv_row.empty else None


def _ci_caption(metric: str, base: str) -> str:
    ci = bootstrap_ci_for(bootstrap_df, CHAMPION, metric) if not bootstrap_df.empty else None
    if ci is None:
        return base
    return f"{base} · 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"


st.caption(f"Shipped model: **{CHAMPION}** (computed by the pre-declared selection rule on the development rows only — see Model insights).")
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Test ROC-AUC", f"{rf_row['roc_auc']:.3f}", _ci_caption("roc_auc", f"n_test={metrics['n_test']}"))
with c2:
    kpi_card("Test PR-AUC", f"{rf_row['pr_auc']:.3f}", _ci_caption("pr_auc", f"fraud rate {metrics['fraud_rate']:.1%}"))
with c3:
    kpi_card("Recall @ threshold", f"{rf_row['recall']:.1%}", _ci_caption("recall", f"threshold {metrics['operating_threshold']:.2f}"))
with c4:
    kpi_card("Precision @ threshold", f"{rf_row['precision']:.1%}", _ci_caption("precision", f"threshold {metrics['operating_threshold']:.2f}"))

if cv_row is not None:
    st.caption(
        f"Repeated CV on the 800 DEVELOPMENT rows only (refit per fold, a different source of uncertainty — how much this moves across "
        f"different TRAINING splits, not just different samples of this one test set): "
        f"F1 {cv_row['f1_mean']:.3f}±{cv_row['f1_std']:.3f} · "
        f"ROC-AUC {cv_row['roc_auc_mean']:.3f}±{cv_row['roc_auc_std']:.3f} · "
        f"recall {cv_row['recall_mean']:.3f}±{cv_row['recall_std']:.3f}. Full table on the Model insights page."
    )

decision = load_champion_decision()
ba = metrics.get("baseline_agreement")
if decision and ba:
    from app.ml.reporting import champion_vs_rule_sentence
    vr = decision["champion_vs_rule"]
    st.markdown(
        f"""<div class="aeg-note" style="border-color:{WARNING}55;background:{WARNING}14;">
        <b>Rule-baseline check (shown on purpose).</b> {champion_vs_rule_sentence(decision)}
        Development CV, champion minus rule: PR-AUC {vr['pr_auc']['mean_difference']:+.3f} (p = {vr['pr_auc']['p_corrected']:.3f}),
        F1 {vr['f1']['mean_difference']:+.3f} (p = {vr['f1']['p_corrected']:.3f}). On the test set the champion's review
        decision matches the rule <i>flag if incident severity = Major Damage</i> for <b>{ba['test_decision_agreement']:.0%}</b>
        of {ba['n_test']} claims. The model's contribution is ranking and per-claim explanation.</div>""",
        unsafe_allow_html=True)

st.write("")
if oracle_results_available():
    oracle = load_oracle_report()
    roc = oracle["oracle_metrics"]["roc_auc"]
    # SH-04: Oracle's own bootstrap CI (evaluate_oracle.py), reported
    # alongside the point estimate exactly like the internal numbers above
    # — 15,420 rows is large, but the collapse itself is worth an interval.
    oracle_roc_ci = (oracle.get("oracle_metrics_ci") or {}).get("roc_auc")
    ci_text = f" (95% CI [{oracle_roc_ci['ci_lower']:.3f}, {oracle_roc_ci['ci_upper']:.3f}])" if oracle_roc_ci else ""
    _short, long_text = oracle_roc_verdict(roc, oracle_roc_ci)
    st.markdown(
        f"""<div class="aeg-note" style="border-color:{DANGER}55;background:{DANGER}14;">
        <b>⚠ External validation warning — deliberately not hidden.</b><br/>
        Scored against Oracle (a real, independently-collected 15,420-row auto-insurance-fraud dataset
        this model never trained on), the model's {long_text}. See the "Monitoring & external validation"
        page for the root cause.</div>""",
        unsafe_allow_html=True,
    )
else:
    st.info("Run `python -m app.ml.evaluate_oracle` from `backend/` to populate external validation results.")

st.write("")
st.markdown("#### Model comparison (internal holdout)")
display_comparison = comparison[["model", "threshold", "recall", "precision", "f1", "pr_auc", "roc_auc", "accuracy"]].copy()
if not bootstrap_df.empty:
    for metric in ("recall", "pr_auc", "roc_auc"):
        display_comparison[f"{metric}_95ci"] = display_comparison.apply(
            lambda row: (lambda ci: f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—")(
                bootstrap_ci_for(bootstrap_df, row["model"], metric)
            ),
            axis=1,
        )
st.dataframe(display_comparison, width="stretch", hide_index=True)
st.caption(
    "`_95ci` columns are bootstrap 95% confidence intervals on this fixed holdout split "
    "(`data/processed/holdout_bootstrap_ci.csv`) — not the same as the cross-fold mean±SD above."
)

if "session_scored_count" in st.session_state:
    st.write("")
    st.caption(f"This session: {st.session_state['session_scored_count']} claim(s) scored via the Score/Batch pages.")
