"""app_pages/model_insights.py — global SHAP importance, the feature audit,
and the generalization findings promoted to a first-class page, not a
buried expander."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.data_access import load_cross_validation, load_shap_importance, models_are_available
from components.theme import ACCENT, DANGER, MUTED, inject_css, page_header

inject_css()
page_header("Model insights", "Global feature importance and the honest feature-quality audit — including the two features flagged as NOT safe to treat as real fraud signal.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

shap_df = load_shap_importance()
cv_df = load_cross_validation()

tab1, tab2, tab3 = st.tabs(["Global feature importance", "Feature quality audit", "Cross-validation"])

with tab1:
    top20 = shap_df.head(20).sort_values("mean_abs_shap")
    risky = {"zip3_risk_tier_low_risk", "zip3_risk_tier_medium_risk", "zip3_risk_tier_high_risk", "is_highrisk_hobby", "is_exec_occupation"}
    top20["flag"] = top20["feature"].isin(risky)
    fig = px.bar(top20, x="mean_abs_shap", y="feature", orientation="h", color="flag",
                 color_discrete_map={True: DANGER, False: ACCENT}, labels={"mean_abs_shap": "mean |SHAP|", "feature": ""})
    fig.update_layout(showlegend=False, height=560, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e8ef")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bars = features flagged in the audit tab as not safe to treat as genuine fraud signal.")

    zip3_share = shap_df[shap_df["feature"].str.startswith("zip3_risk_tier")]["share_of_total"].sum()
    st.markdown(
        f"""<div class="aeg-note" style="border-color:{DANGER}55;background:{DANGER}14;">
        <b>{zip3_share:.1%} of total model weight sits on `zip3_risk_tier`</b> — a target-encoded feature
        built from the exact same 1,000 rows the model trains on (disclosed leakage, not fixed here — see
        the audit tab). This is the single biggest reason performance collapses on external data that has
        no ZIP code to compute this from.</div>""",
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("#### Features NOT safe to treat as genuine fraud signal")
    st.markdown(
        """
- **`zip3_risk_tier_*`** — target-encoded from the training set itself (disclosed leakage, not fixed).
  Real information content partly comes from having seen the label.
- **`is_highrisk_hobby`** — chess (82.6% fraud, n=46) and cross-fit (74.3%, n=35) claimants are
  fraud-flagged far more than every other hobby (17-30%) in this 1,000-row dataset. There is no plausible
  causal fraud mechanism connecting chess to insurance fraud — this reads as a small-sample artifact, and
  leaning on it is close to proxy-discrimination (lifestyle choice → risk score) that real insurance
  regulators scrutinize insurers for.
- **`is_exec_occupation`** — same shape at lower severity (36.8% vs. 24.7% base rate, n=76).

**An earlier version of this rebuild also one-hot-encoded the full `insured_hobbies`/`insured_occupation`/
`auto_make` categories (122 features total).** It cross-validated at ROC-AUC 0.94 but collapsed to
0.59-0.78 on a genuine single holdout split, checked across 6 random seeds — the signature of
high-cardinality overfitting on 800 training rows, not real signal. Those full one-hot blocks were
removed in favor of the single documented flag each — the model shipped here has 74 features, not 122.
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
- `incident_severity_ordinal` / `is_major_damage` — plausible in principle, though the ordinal encoding
  is inferred (flagged TODO-VERIFY in `feature_engineering.py`) rather than confirmed against an original
  source labeling scheme.
        """
    )

with tab3:
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.caption(
        "Full-pipeline 5-fold CV — the ZIP3 lookup, scaler, and classifier are ALL refit per fold, not "
        "reused from a single fit. An earlier version of this file called sklearn's cross_validate() on an "
        "already-engineered matrix and got ROC-AUC ≈0.94 — inflated by exactly the same self-leakage the "
        "audit tab describes, since most rows in each fold had already had their own label baked into "
        "zip3_risk_tier. Refitting per fold (as this table does) closes that leak."
    )
