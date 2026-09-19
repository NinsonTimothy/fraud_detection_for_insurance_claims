"""app_pages/model_insights.py — global SHAP importance, the feature audit,
and the generalization findings promoted to a first-class page, not a
buried expander."""
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
from components.data_access import load_cross_validation, load_shap_importance, models_are_available
from components.theme import ACCENT, DANGER, MUTED, inject_css, page_header

inject_css()
page_header("Model insights", "Global feature importance and the honest feature-quality audit — including the features flagged as NOT safe to treat as real fraud signal.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

shap_df = load_shap_importance()
cv_df = load_cross_validation()

tab1, tab2, tab3 = st.tabs(["Global feature importance", "Feature quality audit", "Cross-validation"])

with tab1:
    top20 = shap_df.head(20).sort_values("mean_abs_shap")
    risky = {"is_highrisk_hobby", "is_exec_occupation"}
    top20["flag"] = top20["feature"].isin(risky)
    fig = px.bar(top20, x="mean_abs_shap", y="feature", orientation="h", color="flag",
                 color_discrete_map={True: DANGER, False: ACCENT}, labels={"mean_abs_shap": "mean |SHAP|", "feature": ""})
    fig.update_layout(showlegend=False, height=560, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e8ef")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bars = features flagged in the audit tab as not safe to treat as genuine fraud signal.")

    st.markdown(
        f"""<div class="aeg-note" style="border-color:{ACCENT}55;background:{ACCENT}14;">
        <b>PB-02 fix (this build):</b> an earlier `zip3_risk_tier` feature — built by taking
        `insured_zip // 100`, which turned out to be a near-unique 4-digit prefix on this dataset's
        6-digit ZIPs, not a genuine 3-digit ZIP3 — has been removed entirely. It let the model memorize
        labels on train (0.2%-71.8% fraud rate by tier) while collapsing to a flat ~20-28% on test, and
        it had absorbed 53.2% of total SHAP weight. It was previously disclosed here as "leakage, kept
        in"; once the actual bug was understood there was no honest version of it to keep, so it's gone
        rather than kept-and-disclosed. See `docs/REBUILD_NOTES.md` for the full evidence.</div>""",
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("#### Features NOT safe to treat as genuine fraud signal")
    st.markdown(
        """
**As of SH-02/D3, both features below are excluded from the deployable model by default** — gated
behind `INCLUDE_PROXY_FEATURES` in `app/core/config.py` (default `False`), not merely disclosed-and-kept.
The table under "Global feature importance" reflects whichever variant this model was actually trained
with; see `data/processed/proxy_feature_ablation.csv` for the measured performance cost of excluding
them (a real drop in recall/ROC-AUC, disclosed rather than hidden — this project's judgment is that
shipping an unexplained lifestyle/occupation → risk association isn't worth that gain).

- **`is_highrisk_hobby`** — chess (82.6% fraud, n=46) and cross-fit (74.3%, n=35) claimants are
  fraud-flagged far more than every other hobby (17-30%) in this 1,000-row dataset. There is no plausible
  causal fraud mechanism connecting chess to insurance fraud — this reads as a small-sample artifact, and
  leaning on it is close to proxy-discrimination (lifestyle choice → risk score) that real insurance
  regulators scrutinize insurers for.
- **`is_exec_occupation`** — same shape at lower severity (36.8% vs. 24.7% base rate, n=76).

**A `zip3_risk_tier_*` feature was removed in this build (PB-02)** — it was a target-encoded lookup built
from `insured_zip // 100`, which turned out to be a near-row-unique 4-digit prefix rather than a genuine
3-digit ZIP3, letting the model memorize training labels. See the Global feature importance tab for the
full evidence; it's no longer part of the shipped feature set.

**An earlier version of this rebuild also one-hot-encoded the full `insured_hobbies`/`insured_occupation`/
`auto_make` categories (122 features total).** It cross-validated at ROC-AUC 0.94 but collapsed to
0.59-0.78 on a genuine single holdout split, checked across 6 random seeds — the signature of
high-cardinality overfitting on 800 training rows, not real signal. Those full one-hot blocks were
removed in favor of the single documented flag each.
        """
    )
    st.write("")
    st.markdown("#### Features that ARE real, defensible signal")
    st.markdown(
        """
- `claim_to_premium_ratio` / `vehicle_claim_pct` / `injury_claim_pct` — claim size relative to premium
  is a standard actuarial red flag.
- `policy_age_at_incident_days` / `is_new_customer` — "new policy, big claim shortly after" is one of
  the best-documented real fraud indicators.
- `is_no_witness` — zero independent witnesses, a weak but genuine signal.
- `incident_severity_ordinal` / `is_major_damage` — the ordinal order (Trivial < Minor < Major < Total
  Loss) is confirmed against the original FYP project's own preprocessing notebook (PB-24), not inferred.
        """
    )

with tab3:
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.caption(
        "Full-pipeline 5-fold CV — the scaler and classifier are refit from scratch on each fold's own "
        "training partition, not reused from a single fit. An earlier version of this file called "
        "sklearn's cross_validate() on an already-engineered matrix and got ROC-AUC ≈0.94 — inflated "
        "because, at the time, most rows in each fold had already had their own label baked into the "
        "zip3_risk_tier feature that has since been removed (PB-02). Refitting per fold (as this table "
        "does) is kept as the honest pattern going forward."
    )
