"""app_pages/monitoring.py — Monitoring & external validation.

The shipped champion, unmodified, scored against Oracle (a real,
independently collected dataset it never trained on). Every number is read
from artifacts; the verdict wording comes from app.ml.reporting.roc_ci_verdict
so it can never go stale (UI-10: says "inverted" only if the whole CI is below
0.5, "better than random" only if it is entirely above, otherwise "no
measurable ranking signal").

UI-11: st.metric deltas are only ever NUMERIC changes (Oracle minus internal),
never descriptive text — a text delta renders a meaningless green arrow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    sys.path.append(_dashboard_root)
from components.charts import _LAYOUT, internal_vs_external, reliability_chart
from components.data_access import (
    PROCESSED_DIR, load_metrics, load_oracle_model_comparison, load_oracle_psi, load_oracle_report,
    models_are_available, oracle_results_available, oracle_roc_verdict,
)
from components.theme import ACCENT, DANGER, MUTED, SUCCESS, inject_css, page_header

inject_css()
page_header("Monitoring & external validation",
            "The shipped model, unmodified, scored against Oracle — a real, independently collected dataset it never trained on.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.run_all` from `backend/` first.")
    st.stop()
if not oracle_results_available():
    st.info("Run `python -m app.ml.evaluate_oracle` from `backend/` to populate this page.")
    st.stop()

report = load_oracle_report()
metrics = load_metrics()
champ = metrics["primary_model"]
champ_row = next(r for r in metrics["model_comparison"] if r["model"] == champ)
internal = {"roc_auc": champ_row["roc_auc"], "pr_auc": champ_row["pr_auc"], "recall": champ_row["recall"]}
oracle = report["oracle_metrics"]
roc_ci = (report.get("oracle_metrics_ci") or {}).get("roc_auc")
short_verdict, long_verdict = oracle_roc_verdict(oracle["roc_auc"], roc_ci)

st.markdown(
    f"""<div class="aeg-note" style="border-color:{DANGER}55;background:{DANGER}14;">
    <b>Oracle result: {short_verdict}.</b> {long_verdict}.
    {report['n_features_constant_on_oracle']} of {report['n_features_total']} trained features
    ({report['share_of_shap_weight_constant_on_oracle']:.1%} of the model's SHAP weight) are constant once Oracle
    data is mapped in: Oracle has no incident-severity field and no claim-amount breakdown, so the features the
    model relies on most are frozen at their defaults. See "Root cause" below.</div>""",
    unsafe_allow_html=True,
)

# ---- UI-11: numeric deltas only (Oracle minus internal), red when negative ----
st.write("")
HELP = "Change relative to the internal 200-row test set."
d_roc = oracle["roc_auc"] - internal["roc_auc"]
d_pr = oracle["pr_auc"] - internal["pr_auc"]
d_rec = oracle["recall"] - internal["recall"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC on Oracle", f"{oracle['roc_auc']:.3f}", f"{d_roc:+.3f} vs internal", delta_color="normal", help=HELP)
c2.metric("PR-AUC on Oracle", f"{oracle['pr_auc']:.3f}", f"{d_pr:+.3f} vs internal", delta_color="normal", help=HELP)
c3.metric("Recall @ review threshold", f"{oracle['recall']:.3f}", f"{d_rec:+.3f} vs internal", delta_color="normal", help=HELP)
with c4:
    st.metric("Oracle rows scored", f"{report['oracle_n_rows']:,}")
    st.caption(f"fraud rate {report['oracle_fraud_rate']:.2%}")

tab1, tab2, tab3 = st.tabs(["Root cause", "Is Oracle learnable at all? (stress test)", "Methodology"])

with tab1:
    const = float(report["share_of_shap_weight_constant_on_oracle"])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["SHAP weight"], x=[1 - const], orientation="h", name="features that vary on Oracle",
                         marker_color=ACCENT, text=[f"varies on Oracle: {1 - const:.1%}"], textposition="inside"))
    fig.add_trace(go.Bar(y=["SHAP weight"], x=[const], orientation="h", name="features constant on Oracle",
                         marker_color=DANGER, text=[f"constant on Oracle: {const:.1%}"], textposition="inside"))
    fig.update_layout(barmode="stack", height=170, xaxis_tickformat=".0%", xaxis_range=[0, 1], showlegend=False,
                      title="Where the model's decision weight sits, once Oracle data is mapped in", **_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{report['n_features_constant_on_oracle']} of {report['n_features_total']} features are constant on Oracle; "
               f"only {report['n_features_variable_on_oracle']} vary. The model's decisions rest almost entirely on fields "
               "Oracle never collected.")
    ext = PROCESSED_DIR.parent / "external" / "oracle"
    with st.expander("Field mapping (raw field → Oracle source)"):
        p = ext / "oracle_field_mapping.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p), width="stretch", hide_index=True)
    with st.expander("Univariate AUC of the features that still vary (development vs Oracle)"):
        p = ext / "oracle_univariate_auc.csv"
        if p.exists():
            st.dataframe(pd.read_csv(p).drop(columns=["direction_reverses"], errors="ignore").round(3), width="stretch", hide_index=True)
            st.caption("AUCs near 0.5 on the development data mean these features carried almost no signal to transfer.")

with tab2:
    stress_df = load_oracle_model_comparison()
    if len(stress_df):
        st.markdown("A **different question**: fresh models trained directly on Oracle's own fields (not this project's model):")
        st.dataframe(stress_df.round(3), width="stretch", hide_index=True)
        best = stress_df.loc[stress_df["roc_auc"].idxmax()]
        st.markdown(
            f"""<div class="aeg-note" style="background:{SUCCESS}14;border-color:{SUCCESS}55;">
            Oracle is learnable fraud data: the best fresh model (<code>{best['model']}</code>) reaches ROC-AUC
            {best['roc_auc']:.3f} on Oracle's own fields. The collapse above is a <b>feature-availability problem</b>,
            not evidence that Oracle is unlearnable.</div>""", unsafe_allow_html=True)

with tab3:
    counts = report.get("field_mapping_counts", {})
    st.markdown("**Field mapping:** " + ", ".join(f"{v} {k}" for k, v in counts.items()) +
                ". Every unmapped raw field falls back to its documented default; nothing is invented.")
    psi_df = load_oracle_psi()
    if len(psi_df):
        st.markdown("**Population Stability Index** on the fields Oracle genuinely supplies:")
        st.dataframe(psi_df.round(3), width="stretch", hide_index=True)
        st.caption("PSI > 0.2 is conventionally significant drift. PSI is not computed for constant features (it is meaningless there).")
    st.caption(f"Model: {report.get('model')} · generated {report['generated_at'][:19].replace('T', ' ')} UTC · "
               "regenerate with `python -m app.ml.evaluate_oracle`.")

st.divider()
st.markdown("### Internal vs external validation")
ci = pd.read_csv(PROCESSED_DIR / "holdout_bootstrap_ci.csv")
row = ci[(ci["model"] == champ) & (ci["metric"] == "roc_auc")]
if len(row) and roc_ci:
    st.plotly_chart(internal_vs_external([
        {"label": f"Internal test ({metrics['n_test']})", "roc_auc": float(row["point_estimate"].iloc[0]),
         "lo": float(row["ci_lower"].iloc[0]), "hi": float(row["ci_upper"].iloc[0])},
        {"label": f"Oracle ({report['oracle_n_rows']:,})", "roc_auc": oracle["roc_auc"], "lo": roc_ci["ci_lower"], "hi": roc_ci["ci_upper"]},
    ]), width="stretch")

st.markdown("### Calibration (champion, test set)")
rel_path = PROCESSED_DIR / "reliability_curves.csv"
if rel_path.exists():
    rel = pd.read_csv(rel_path)
    st.plotly_chart(reliability_chart(rel[rel["model"] == champ], "test", champ), width="stretch")
    cs = pd.read_csv(PROCESSED_DIR / "calibration_summary.csv")
    cs = cs[(cs["split"] == "test") & (cs["model"] != "major_damage_rule")]
    table = cs[["model", "brier", "ece"]].copy()
    table.loc[len(table)] = {"model": "reference: always predict the base rate", "brier": float(cs["brier_base_rate"].iloc[0]), "ece": None}
    st.dataframe(table.round(3), width="stretch", hide_index=True)
    st.caption(f"Brier: lower is better. ECE: mean gap between predicted and observed fraud rate over 10 bins. "
               f"The champion ships with {metrics['calibration']['chosen']} calibration chosen on development data.")
