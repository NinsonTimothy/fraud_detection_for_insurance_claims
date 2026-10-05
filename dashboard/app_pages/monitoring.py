"""app_pages/monitoring.py — external validation (Oracle) + PSI drift,
the generalization-testing story front and center rather than buried."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    # PB-01: append, never insert(0, ...) — inserting the dashboard
    # dir at the FRONT of sys.path on every rerun is what let
    # `import app...` resolve to the old dashboard/app.py instead of
    # the backend's app/ package (see components/data_access.py,
    # which puts backend/ at sys.path[0] once, on first import).
    sys.path.append(_dashboard_root)
from components.data_access import (
    load_oracle_model_comparison, load_oracle_psi, load_oracle_report,
    models_are_available, oracle_results_available, oracle_roc_verdict,
)
from components.theme import ACCENT, DANGER, MUTED, SUCCESS, WARNING, inject_css, page_header

inject_css()
page_header("Monitoring & external validation", "The live model, unmodified, scored against Oracle — a real, independently-collected dataset it never trained on.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

if not oracle_results_available():
    st.info("Run `python -m app.ml.evaluate_oracle` from `backend/` to populate this page.")
    st.stop()

report = load_oracle_report()
internal = report["internal_holdout_metrics"]
oracle = report["oracle_metrics"]
# OR-01: the headline wording is derived from the bootstrap CI, never from
# a fixed "within 0.05 of 0.5" rule. With a CI entirely below 0.5 the
# correct description is "significantly inverted", not "random".
short_verdict, long_verdict = oracle_roc_verdict(oracle["roc_auc"], (report.get("oracle_metrics_ci") or {}).get("roc_auc"))

st.markdown(
    f"""<div class="aeg-note" style="border-color:{DANGER}55;background:{DANGER}14;">
    <b>Oracle result: {short_verdict}.</b> The {long_verdict}.
    ROC-AUC is {internal['roc_auc']:.3f} on the internal holdout.
    {report['n_features_constant_on_oracle']} of {report['n_features_total']} trained features
    ({report['share_of_shap_weight_constant_on_oracle']:.1%} of total SHAP weight) go completely
    constant once Oracle-mapped data passes through — Oracle has no incident-severity field and no
    claim-dollar breakdown, so the model's heaviest-weighted features are frozen at their defaults and
    the few fields that do vary decide the ranking. A likely reason it is inverted rather than random:
    witnesses relate to fraud in the OPPOSITE direction on Oracle (a witness present: 3.4% fraud vs. 6.0%
    without) to this project's training data (more witnesses, slightly more fraud — a dataset artefact).
    See "Root cause" below.</div>""",
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC — Oracle vs. internal", f"{oracle['roc_auc']:.3f}", f"internal {internal['roc_auc']:.3f}")
c2.metric("PR-AUC — Oracle vs. internal", f"{oracle['pr_auc']:.3f}", f"internal {internal['pr_auc']:.3f}")
c3.metric("Recall @ operating threshold", f"{oracle['recall']:.1%}")
c4.metric("Oracle rows scored", f"{report['oracle_n_rows']:,}", f"fraud rate {report['oracle_fraud_rate']:.2%}")

tab1, tab2, tab3 = st.tabs(["Root cause (feature drift)", "Is Oracle learnable at all? (stress test)", "Methodology"])

with tab1:
    psi_df = load_oracle_psi()
    if len(psi_df):
        fig = px.bar(psi_df.sort_values("psi"), x="psi", y="feature", orientation="h",
                     color=psi_df.sort_values("psi")["significant_drift"],
                     color_discrete_map={True: DANGER, False: ACCENT}, labels={"psi": "PSI"})
        fig.add_vline(x=0.2, line_dash="dash", line_color=WARNING, annotation_text="critical (0.2)")
        fig.update_layout(showlegend=False, height=max(300, 26 * len(psi_df)), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e8ef")
        st.plotly_chart(fig, width="stretch")
    st.caption("Population Stability Index on the few fields Oracle genuinely supplies (age, sex, deductible, witnesses, vehicle count/year). Every other feature this model relies on is constant on Oracle by construction — not shown here because a PSI on a constant is meaningless, not because it's fine.")

with tab2:
    stress_df = load_oracle_model_comparison()
    if len(stress_df):
        st.markdown("Fresh models trained **directly on Oracle's own real fields** (not this project's model):")
        st.dataframe(stress_df, width="stretch", hide_index=True)
        st.markdown(
            f"""<div class="aeg-note" style="background:{SUCCESS}14;border-color:{SUCCESS}55;">
            Oracle IS learnable fraud data — a fresh XGBoost model reaches
            ROC-AUC {stress_df.set_index('model').loc['oracle_xgb','roc_auc']:.3f} when trained on
            Oracle's own fields (Fault, PastNumberOfClaims, etc.). This confirms the collapse above is a
            <b>feature-availability problem</b>, not evidence Oracle is unlearnable — the shipped model was
            simply never built on fields that would survive a change of data source.</div>""",
            unsafe_allow_html=True,
        )

with tab3:
    st.markdown(f"**Fields mapped from Oracle for real:** `{'`, `'.join(report['fields_mapped_for_real'])}`")
    st.caption("Every other raw field falls back to its documented default (feature_engineering.MISSING_COLUMN_DEFAULTS) — nothing invented to move the score either way, same honest-mapping rule as the sibling MoMo Guard project's PaySim adapter.")
    st.write("")
    st.caption(f"Model: {report.get('model', 'random_forest')} · Generated {report['generated_at'][:19].replace('T',' ')} UTC · Regenerate with `python -m app.ml.evaluate_oracle`.")


# ---------------------------------------------------------------- C1 / B8 ---
import json as _json
import pandas as pd

from components.charts import internal_vs_external, reliability_chart
from components.data_access import PROCESSED_DIR as _P, load_metrics

st.divider()
st.markdown("### Calibration (champion and comparators)")
_rel_path = _P / "reliability_curves.csv"
if _rel_path.exists():
    _rel = pd.read_csv(_rel_path)
    _champ = load_metrics().get("primary_model")
    r1, r2 = st.columns(2)
    r1.plotly_chart(reliability_chart(_rel, "development_oof", _champ), width="stretch")
    r2.plotly_chart(reliability_chart(_rel, "test", _champ), width="stretch")
    st.dataframe(pd.read_csv(_P / "calibration_summary.csv").round(3), width="stretch", hide_index=True)
    st.caption("Brier: lower is better (the base-rate row is what always predicting the fraud rate scores). "
               "ECE: mean gap between predicted and observed fraud rate across 10 bins.")

st.markdown("### Internal vs external validation")
_ci = pd.read_csv(_P / "holdout_bootstrap_ci.csv")
_champ = load_metrics().get("primary_model")
_row = _ci[(_ci["model"] == _champ) & (_ci["metric"] == "roc_auc")]
_oc = report.get("oracle_metrics_ci", {}).get("roc_auc")
if len(_row) and _oc:
    st.plotly_chart(internal_vs_external([
        {"label": "Internal test (200)", "roc_auc": float(_row["point_estimate"].iloc[0]), "lo": float(_row["ci_lower"].iloc[0]), "hi": float(_row["ci_upper"].iloc[0])},
        {"label": f"Oracle ({report['oracle_n_rows']:,})", "roc_auc": report["oracle_metrics"]["roc_auc"], "lo": _oc["ci_lower"], "hi": _oc["ci_upper"]},
    ]), width="stretch")
_ext = _P.parent / "external" / "oracle"
if (_ext / "oracle_field_mapping.csv").exists():
    st.markdown("**Field mapping (raw field → Oracle source)**")
    st.dataframe(pd.read_csv(_ext / "oracle_field_mapping.csv"), width="stretch", hide_index=True)
if (_ext / "oracle_univariate_auc.csv").exists():
    st.markdown("**Univariate AUC of the features that still vary on Oracle (development vs Oracle)**")
    st.dataframe(pd.read_csv(_ext / "oracle_univariate_auc.csv").round(3), width="stretch", hide_index=True)
    st.caption("AUCs near 0.5 on the development data mean these features carried almost no signal to transfer in "
               "the first place; a reversal flag means the relationship flips between datasets.")
