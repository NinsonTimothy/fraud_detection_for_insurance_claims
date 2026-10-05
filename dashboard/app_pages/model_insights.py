"""app_pages/model_insights.py — global SHAP importance, the feature audit,
and the generalization findings promoted to a first-class page, not a
buried expander."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_dashboard_root = str(Path(__file__).resolve().parents[1])
if _dashboard_root not in sys.path:
    # PB-01: append, never insert(0, ...) — inserting the dashboard
    # dir at the FRONT of sys.path on every rerun is what let
    # `import app...` resolve to the old dashboard/app.py instead of
    # the backend's app/ package (see components/data_access.py,
    # which puts backend/ at sys.path[0] once, on first import).
    sys.path.append(_dashboard_root)
from components.data_access import PROCESSED_DIR, load_cleaned_data, load_witness_fraud_rates, load_champion_decision, load_cross_validation, load_field_importance, load_metrics, load_pairwise_tests, load_selection_summary, load_shap_importance, models_are_available
from components.charts import GRADE_COLOURS, _LAYOUT
from components.theme import ACCENT, DANGER, MUTED, WARNING, inject_css, page_header

inject_css()
page_header("Model insights", "Global feature importance and the honest feature-quality audit — including the features flagged as NOT safe to treat as real fraud signal.")

if not models_are_available():
    st.warning("No trained model found. Run `python -m app.ml.train` from `backend/` first.")
    st.stop()

shap_df = load_shap_importance()
cv_df = load_cross_validation()

# UI-13: the old "Cross-validation" tab duplicated the selection table; merged into tab0.
tab0, tab1, tab2 = st.tabs(["Model selection & baseline", "Global feature importance", "Feature quality audit"])

with tab0:
    decision = load_champion_decision()
    summary = load_selection_summary()
    tests = load_pairwise_tests()
    metrics = load_metrics()
    if not decision or summary.empty:
        st.info("Run `python -m app.ml.run_all` to produce the model-selection artefacts.")
    else:
        from app.ml.reporting import champion_vs_rule_sentence
        champ = decision["champion"]
        st.markdown("**Protocol (pre-declared, applied mechanically):**\n" + "\n".join(
            f"- *{k.replace('_', ' ')}:* {v}" for k, v in decision["protocol"].items()))
        st.markdown(f"**Champion (computed): `{champ}`** · best PR-AUC model: `{decision['best_pr_auc_model']}`")
        for t in decision["decision_trail"]:
            st.markdown(f"- `{t['candidate']}` — {t['decision']}" + (f" (p = {t['vs_best_p']:.3f})" if "vs_best_p" in t else ""))
        st.markdown(f"<div class='aeg-note' style='border-color:{WARNING}55;background:{WARNING}14;'>"
                    f"{champion_vs_rule_sentence(decision)}</div>", unsafe_allow_html=True)

        # UI-13: neutral grey, champion in the accent colour, no red for any model
        show = summary.sort_values("pr_auc_mean", ascending=False)
        fig = go.Figure(go.Bar(
            x=show["model"], y=show["pr_auc_mean"], marker_color=[ACCENT if m == champ else "#6b7280" for m in show["model"]],
            error_y=dict(type="data", array=show["pr_auc_std"]), text=[f"{v:.3f}" for v in show["pr_auc_mean"]], textposition="outside"))
        fig.update_layout(height=320, yaxis_title="PR-AUC (development CV, mean ± SD)", **_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        cvt = pd.DataFrame({"model": show["model"], **{m: [f"{a:.3f} ± {b:.3f}" for a, b in zip(show[f"{m}_mean"], show[f"{m}_std"])]
                                                     for m in ("pr_auc", "roc_auc", "f1", "recall")}})
        st.dataframe(cvt, width="stretch", hide_index=True)
        st.caption(f"Repeated stratified CV on the {metrics['n_train']} development rows only; models refit on every fold.")
        if not tests.empty:
            mine = tests[((tests["a"] == champ) | (tests["b"] == champ)) & tests["metric"].isin(["pr_auc", "f1"])].copy()
            mine["vs"] = [b if a == champ else a for a, b in zip(mine["a"], mine["b"])]
            mine["champion_minus_other"] = [d if a == champ else -d for a, d in zip(mine["a"], mine["mean_difference"])]
            st.markdown("**Champion vs each other model (corrected resampled t-test):**")
            st.dataframe(mine[["vs", "metric", "champion_minus_other", "p_corrected", "significant"]].round(4),
                         width="stretch", hide_index=True)
            with st.expander("All pairwise tests"):
                st.dataframe(tests.round(4), width="stretch", hide_index=True)

    probe_path = PROCESSED_DIR / "sensitivity_probe.csv"
    if probe_path.exists():
        st.markdown("#### Sensitivity probe — one field changed at a time on a reference claim")
        probe = pd.read_csv(probe_path)
        probe = probe[probe["field"].isin(["incident_severity", "witnesses", "police_report_available"])].copy()
        probe["label"] = probe["field"].map({"incident_severity": "Severity", "witnesses": "Witnesses",
                                             "police_report_available": "Police report"}) + ": " + probe["value"].astype(str)
        thr = load_metrics()["operating_threshold"]
        fig = go.Figure(go.Bar(x=probe["label"], y=probe["fraud_risk_score"],
                               marker_color=[GRADE_COLOURS[b] for b in probe["risk_band"]],
                               text=[f"{v:.2f} · {b}" for v, b in zip(probe["fraud_risk_score"], probe["risk_band"])],
                               textposition="outside"))
        fig.add_hline(y=thr, line_dash="dash", line_color=MUTED)
        fig.add_annotation(x=1.0, xref="paper", y=thr, text=f"review threshold {thr:.2f}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=11, color=MUTED))
        fig.update_layout(height=380, yaxis_title="Fraud-risk score", yaxis_range=[0, max(1.0, probe["fraud_risk_score"].max() + 0.1)], **{**_LAYOUT, "margin": dict(l=10, r=140, t=30, b=10)})
        st.plotly_chart(fig, width="stretch")
        st.caption("Bar colour = risk band (green Low, amber Medium, red High); the label repeats the band in text.")
        st.caption("Severity: Total Loss scores well below Major Damage, close to Minor Damage, because Total Loss "
                   "claims are rarely fraud in this dataset. Disclosed in docs/LIMITATIONS.md.")
        rates = load_witness_fraud_rates()
        st.caption("Witnesses: in the training data the fraud rate rises with witnesses ("
                   + ", ".join(f"{k}: {v:.1%}" for k, v in rates.items())
                   + "); the model reflects this dataset artefact, disclosed in docs/LIMITATIONS.md.")

with tab1:
    # A7: same parent-field aggregation as the per-claim explanations.
    fimp = load_field_importance()
    top20 = fimp.head(10).sort_values("mean_abs_shap")  # UI-13: top 10 parent fields
    top20["flag"] = top20["source_field"].isin({"insured_hobbies", "insured_occupation"})
    fig = px.bar(top20, x="share_of_total", y="display_name", orientation="h", color="flag",
                 color_discrete_map={True: DANGER, False: ACCENT}, labels={"share_of_total": "share of global mean |SHAP|", "display_name": ""})
    fig.update_layout(showlegend=False, height=400, xaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e8ef")
    st.plotly_chart(fig, width="stretch")
    st.caption("One bar per claim FIELD: one-hot columns and single-parent engineered features (e.g. is_major_damage "
               "and the severity ordinal -> Incident severity) are summed, exactly as in each claim's explanation.")
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
    # UI-16: every rate below is computed from the cleaned training data, never typed.
    _df = load_cleaned_data()
    _y = _df["fraud_reported"].eq("Y")
    _base = float(_y.mean())
    _hob = _df.assign(y=_y).groupby("insured_hobbies")["y"].agg(["mean", "size"])
    _h = {k: (float(_hob.loc[k, "mean"]), int(_hob.loc[k, "size"])) for k in ("chess", "cross-fit")}
    _rest = _hob.drop(index=["chess", "cross-fit"])["mean"]
    _h_lo, _h_hi = float(_rest.min()), float(_rest.max())
    _occ = _df.assign(y=_y).groupby("insured_occupation")["y"].agg(["mean", "size"])
    _o = (float(_occ.loc["exec-managerial", "mean"]), int(_occ.loc["exec-managerial", "size"]))
    _w = ", ".join(f"{k}: {v:.1%}" for k, v in load_witness_fraud_rates().items())
    st.markdown("#### Features NOT safe to treat as genuine fraud signal")
    st.markdown(
        f"""
**As of SH-02/D3, both features below are excluded from the deployable model by default** — gated
behind `INCLUDE_PROXY_FEATURES` in `app/core/config.py` (default `False`), not merely disclosed-and-kept.
The table under "Global feature importance" reflects whichever variant this model was actually trained
with; see `data/processed/proxy_feature_ablation.csv` for the measured performance cost of excluding
them (a real drop in recall/ROC-AUC, disclosed rather than hidden — this project's judgment is that
shipping an unexplained lifestyle/occupation → risk association isn't worth that gain).

- **`is_highrisk_hobby`** — chess ({_h['chess'][0]:.1%} fraud, n={_h['chess'][1]}) and cross-fit ({_h['cross-fit'][0]:.1%}, n={_h['cross-fit'][1]}) claimants are
  fraud-flagged far more than every other hobby ({_h_lo:.0%}–{_h_hi:.0%}) in this 1,000-row dataset. There is no plausible
  causal fraud mechanism connecting chess to insurance fraud — this reads as a small-sample artifact, and
  leaning on it is close to proxy-discrimination (lifestyle choice → risk score) that real insurance
  regulators scrutinize insurers for.
- **`is_exec_occupation`** — same shape at lower severity ({_o[0]:.1%} vs. {_base:.1%} base rate, n={_o[1]}).

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
        f"""
- `claim_to_premium_ratio` / `vehicle_claim_pct` / `injury_claim_pct` — claim size relative to premium
  is a standard actuarial red flag.
- `policy_age_at_incident_days` / `is_new_customer` — "new policy, big claim shortly after" is one of
  the best-documented real fraud indicators.
- **`is_no_witness` — REMOVED from the model (MS-02).** "No witnesses = suspicious" is a real-world red flag, but in
  this dataset zero-witness claims have the LOWEST fraud rate ({_w}).
  The flag therefore contradicted its own name. The raw witness count is still a feature; the pattern is
  documented in docs/LIMITATIONS.md as a dataset artefact, not real fraud behaviour.
- `incident_severity_ordinal` / `is_major_damage` — the ordinal order (Trivial < Minor < Major < Total
  Loss) is confirmed against the original FYP project's own preprocessing notebook (PB-24), not inferred.
        """
    )
