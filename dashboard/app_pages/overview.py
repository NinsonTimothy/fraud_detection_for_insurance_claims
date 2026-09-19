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
from components.data_access import load_metrics, load_model_comparison, models_are_available, oracle_results_available, load_oracle_report
from components.theme import inject_css, kpi_card, page_header, MUTED, DANGER

inject_css()
page_header("Overview", "Real, reproducible metrics from the last training run — nothing here is a fabricated production number.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

metrics = load_metrics()
comparison = load_model_comparison()
rf_row = comparison[comparison["model"] == "random_forest"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Internal test ROC-AUC", f"{rf_row['roc_auc']:.3f}", f"n_test={metrics['n_test']}")
with c2:
    kpi_card("Internal test PR-AUC", f"{rf_row['pr_auc']:.3f}", f"fraud rate {metrics['fraud_rate']:.1%}")
with c3:
    kpi_card("Recall @ operating threshold", f"{rf_row['recall']:.1%}", f"threshold {metrics['operating_threshold']:.2f}")
with c4:
    kpi_card("Precision @ operating threshold", f"{rf_row['precision']:.1%}", f"{metrics['n_features']} engineered features")

st.write("")
if oracle_results_available():
    oracle = load_oracle_report()
    roc = oracle["oracle_metrics"]["roc_auc"]
    st.markdown(
        f"""<div class="aeg-note" style="border-color:{DANGER}55;background:{DANGER}14;">
        <b>⚠ External validation warning — deliberately not hidden.</b><br/>
        Scored against Oracle (a real, independently-collected 15,420-row auto-insurance-fraud dataset
        this model never trained on), ROC-AUC drops to <b>{roc:.3f}</b> — statistically indistinguishable
        from random ({'≈0.50' if abs(roc-0.5) < 0.05 else ''}). See the "Monitoring & external validation"
        page for the full breakdown and root cause.</div>""",
        unsafe_allow_html=True,
    )
else:
    st.info("Run `python -m app.ml.evaluate_oracle` from `backend/` to populate external validation results.")

st.write("")
st.markdown("#### Model comparison (internal holdout)")
st.dataframe(comparison[["model", "threshold", "recall", "precision", "f1", "pr_auc", "roc_auc", "accuracy"]], use_container_width=True, hide_index=True)

if "session_scored_count" in st.session_state:
    st.write("")
    st.caption(f"This session: {st.session_state['session_scored_count']} claim(s) scored via the Score/Batch pages.")
