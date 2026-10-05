"""components/charts.py — reusable Plotly visuals shared by the Score a
claim and Batch review pages, so both explain a score the same way.

Accessibility (WCAG 1.4.1): colour is never the only signal. Every
risk-coloured element also carries a text label ("Low"/"Medium"/"High",
"raises risk"/"lowers risk")."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.theme import ACCENT, DANGER, MUTED, SUCCESS, TEXT, WARNING

GRADE_COLOURS = {"Low": SUCCESS, "Medium": WARNING, "High": DANGER}
GRADE_ORDER = ["Low", "Medium", "High"]
_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT,
               margin=dict(l=10, r=10, t=40, b=10))


def reasons_bar(reasons: list[dict], title: str = "What drove this score") -> go.Figure:
    """Horizontal SHAP contribution chart: one bar per claim FIELD (one-hot
    dummies already grouped by the explainer), labelled with the claim's
    actual value. Red = raises the risk score, green = lowers it."""
    if not reasons:
        return go.Figure()
    df = pd.DataFrame(reasons).iloc[::-1]  # biggest at the top
    labels = [f"{r['display_name']}: {r.get('display_value', r['value'])}" for _, r in df.iterrows()]
    colours = [DANGER if v > 0 else SUCCESS for v in df["shap_value"]]
    text = ["▲ raises risk" if v > 0 else "▼ lowers risk" for v in df["shap_value"]]
    fig = go.Figure(go.Bar(
        x=df["shap_value"], y=labels, orientation="h", marker_color=colours,
        text=text, textposition="auto",
        hovertemplate="%{y}<br>contribution %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(title=title, height=max(280, 42 * len(df)), xaxis_title="Contribution to the score (SHAP)",
                      yaxis_title="", showlegend=False, **_LAYOUT)
    return fig


def probability_histogram(df: pd.DataFrame, threshold: float, edges: dict | None = None) -> go.Figure:
    """Score distribution with the THREE separately-labelled lines (UI-15):
    the two band edges (from the risk-policy artifact) and the review
    threshold. Labels sit at staggered heights so they never overlap."""
    fig = go.Figure()
    for g in GRADE_ORDER:
        sub = df[df["risk_grade"] == g]
        if len(sub):
            fig.add_trace(go.Histogram(x=sub["fraud_probability"], name=g, marker_color=GRADE_COLOURS[g],
                                       xbins=dict(start=0, end=1, size=0.04)))
    lines = []
    if edges:
        lines += [(edges["medium_edge"], f"Low|Medium {edges['medium_edge']:.2f}", MUTED, "dot"),
                  (edges["high_edge"], f"Medium|High {edges['high_edge']:.2f}", MUTED, "dot")]
    lines.append((threshold, f"review threshold {threshold:.2f}", ACCENT, "dash"))
    for i, (x, label, colour, dash) in enumerate(lines):
        fig.add_vline(x=x, line_dash=dash, line_color=colour, line_width=2)
        fig.add_annotation(x=x, y=1.0 - 0.11 * i, yref="paper", text=label, showarrow=False, xanchor="left",
                           xshift=4, font=dict(size=11, color=colour), bgcolor="rgba(11,15,25,0.75)")
    fig.update_layout(title="Fraud-risk score distribution", barmode="stack", height=320,
                      xaxis_title="Fraud-risk score", yaxis_title="Claims", xaxis_range=[0, 1], **_LAYOUT)
    return fig


def flag_rate_by(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    g = df.groupby(column).agg(flag_rate=("flagged", "mean"), n=("flagged", "size")).reset_index()
    g = g.sort_values("flag_rate")
    fig = go.Figure(go.Bar(
        x=g["flag_rate"], y=g[column].astype(str), orientation="h", marker_color=ACCENT,
        text=[f"{r:.0%} of {n}" for r, n in zip(g["flag_rate"], g["n"])], textposition="auto",
    ))
    fig.update_layout(title=title, height=max(260, 34 * len(g)), xaxis_tickformat=".0%",
                      xaxis_title="Share flagged for review", yaxis_title="", **_LAYOUT)
    return fig


def amount_vs_probability(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for g in GRADE_ORDER:
        sub = df[df["risk_grade"] == g]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["total_claim_amount"], y=sub["fraud_probability"], mode="markers", name=g,
                marker=dict(color=GRADE_COLOURS[g], size=7, opacity=0.75),
                customdata=sub["claim_id"] if "claim_id" in sub else None,
                hovertemplate="claim %{customdata}<br>$%{x:,.0f}<br>p=%{y:.2f}<extra>" + g + "</extra>",
            ))
    fig.add_hline(y=threshold, line_dash="dash", line_color=ACCENT)
    fig.update_layout(title="Claim amount vs. fraud-risk score", height=260, xaxis_title="Total claim amount ($)",
                      yaxis_title="Fraud-risk score", yaxis_range=[0, 1], **_LAYOUT)
    return fig


def driver_frequency_bar(reasons_per_claim: list[list[dict]], top_n: int = 10) -> go.Figure:
    """Across a set of (flagged) claims: which fields most often appear as a
    top-3 factor that RAISED the score. Tells a supervisor what is driving
    the queue as a whole, not just one claim."""
    counts: dict[str, int] = {}
    for reasons in reasons_per_claim:
        for r in (reasons or [])[:3]:
            if r.get("shap_value", 0) > 0:
                counts[r["display_name"]] = counts.get(r["display_name"], 0) + 1
    if not counts:
        return go.Figure()
    s = pd.Series(counts).sort_values().tail(top_n)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=DANGER,
                           text=s.values, textposition="auto"))
    fig.update_layout(title="Most common risk-raising factors (top-3 per flagged claim)",
                      height=max(260, 32 * len(s)), xaxis_title="Flagged claims", yaxis_title="", **_LAYOUT)
    return fig


def field_driver_bar(field_shap: pd.DataFrame, labels: dict, top_n: int = 8) -> go.Figure:
    """A1: mean |SHAP| per PARENT raw field across the claims in view."""
    s = field_shap.abs().mean().sort_values().tail(top_n)
    fig = go.Figure(go.Bar(x=s.values, y=[labels.get(i, i) for i in s.index], orientation="h", marker_color=ACCENT,
                           text=[f"{v:.3f}" for v in s.values], textposition="auto"))
    fig.update_layout(title="Top drivers in this view (mean |SHAP| per claim field)", height=max(300, 38 * len(s)),
                      xaxis_title="Mean |SHAP contribution|", yaxis_title="", **_LAYOUT)
    return fig


def score_by_category(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    g = df.groupby(column).agg(mean=("fraud_probability", "mean"), n=("fraud_probability", "size")).reset_index().sort_values("mean")
    fig = go.Figure(go.Bar(x=g["mean"], y=g[column].astype(str), orientation="h", marker_color=WARNING,
                           text=[f"{m:.2f} (n={n})" for m, n in zip(g["mean"], g["n"])], textposition="auto"))
    fig.update_layout(title=title, height=max(240, 40 * len(g)), xaxis_title="Mean fraud-risk score", yaxis_title="",
                      xaxis_range=[0, 1], **_LAYOUT)
    return fig


def reliability_chart(rel: pd.DataFrame, split: str, champion: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED), name="perfect"))
    for model, g in rel[(rel["split"] == split) & (rel["n"] > 0)].groupby("model"):
        fig.add_trace(go.Scatter(x=g["mean_predicted"], y=g["observed_fraud_rate"], mode="lines+markers",
                                 name=f"{model}{' (champion)' if model == champion else ''}",
                                 line=dict(width=4 if model == champion else 1.5),
                                 customdata=g["n"], hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>n=%{customdata}"))
    fig.update_layout(title=f"Reliability ({split.replace('_', ' ')})", height=340, xaxis_title="Mean predicted score",
                      yaxis_title="Observed fraud rate", xaxis_range=[0, 1], yaxis_range=[0, 1], **_LAYOUT)
    return fig


def internal_vs_external(rows: list[dict]) -> go.Figure:
    """rows: {label, roc_auc, lo, hi} — point estimates with 95% CIs."""
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=[r["roc_auc"] for r in rows], marker_color=[ACCENT, DANGER][:len(rows)],
                           error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["roc_auc"] for r in rows],
                                        arrayminus=[r["roc_auc"] - r["lo"] for r in rows]),
                           text=[f"{r['roc_auc']:.3f}" for r in rows], textposition="outside"))
    fig.add_hline(y=0.5, line_dash="dash", line_color=MUTED, annotation_text="random ranking (0.5)")
    fig.update_layout(title="ROC-AUC with 95% bootstrap CI: internal test vs Oracle", height=340,
                      yaxis_range=[0.3, 1.0], yaxis_title="ROC-AUC", **_LAYOUT)
    return fig
