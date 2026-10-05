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


def risk_gauge(probability: float, threshold: float, medium_edge: float, high_edge: float) -> go.Figure:
    """Fraud probability as a 0-100 gauge, banded Low/Medium/High, with the
    review threshold drawn as a marker line."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Fraud probability", "font": {"size": 14, "color": MUTED}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": TEXT, "thickness": 0.25},
            "steps": [
                {"range": [0, medium_edge * 100], "color": f"{SUCCESS}55"},
                {"range": [medium_edge * 100, high_edge * 100], "color": f"{WARNING}55"},
                {"range": [high_edge * 100, 100], "color": f"{DANGER}55"},
            ],
            "threshold": {"line": {"color": ACCENT, "width": 4}, "thickness": 0.9, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=260, **_LAYOUT)
    return fig


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


def score_context_histogram(reference_scores: np.ndarray, this_score: float, threshold: float) -> go.Figure:
    """Where this claim sits relative to the scores of the held-out test
    claims (the model's 'normal' output distribution)."""
    fig = go.Figure(go.Histogram(x=reference_scores, nbinsx=25, marker_color=f"{ACCENT}99", name="Test claims"))
    fig.add_vline(x=threshold, line_dash="dash", line_color=WARNING,
                  annotation_text="review threshold", annotation_position="top left")
    fig.add_vline(x=this_score, line_color=DANGER, line_width=3,
                  annotation_text="this claim", annotation_position="top right")
    fig.update_layout(title="This claim vs. 200 held-out test claims", height=260,
                      xaxis_title="Fraud probability", yaxis_title="Claims", xaxis_range=[0, 1],
                      showlegend=False, **_LAYOUT)
    return fig


def claim_breakdown_donut(injury: float, prop: float, vehicle: float) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Injury", "Property", "Vehicle"], values=[injury, prop, vehicle], hole=0.55,
        marker_colors=[DANGER, WARNING, ACCENT], textinfo="label+percent",
    ))
    fig.update_layout(title="Claim amount breakdown", height=260, showlegend=False, **_LAYOUT)
    return fig


def grade_distribution_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=[GRADE_COLOURS[g] for g in counts.index],
        text=[f"{g}: {n}" for g, n in zip(counts.index, counts.values)], textposition="auto",
    ))
    fig.update_layout(title="Claims by risk grade", height=300, yaxis_title="Claims", **_LAYOUT)
    return fig


def probability_histogram(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for g in GRADE_ORDER:
        sub = df[df["risk_grade"] == g]
        if len(sub):
            fig.add_trace(go.Histogram(x=sub["fraud_probability"], name=g, marker_color=GRADE_COLOURS[g],
                                       xbins=dict(start=0, end=1, size=0.04)))
    fig.add_vline(x=threshold, line_dash="dash", line_color=ACCENT, annotation_text=f"review threshold {threshold:.2f}")
    fig.update_layout(title="Fraud probability distribution", barmode="stack", height=300,
                      xaxis_title="Fraud probability", yaxis_title="Claims", xaxis_range=[0, 1], **_LAYOUT)
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
    fig.update_layout(title="Claim amount vs. fraud probability", height=320, xaxis_title="Total claim amount ($)",
                      yaxis_title="Fraud probability", yaxis_range=[0, 1], **_LAYOUT)
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
