"""
psi.py — Population Stability Index, reused from the same methodology as
the sibling MoMo Guard project's `src/monitoring/psi.py`. Used both for the
live "Monitoring" dashboard page (drift of incoming scored claims vs. the
training distribution) and the Oracle external-validation report.

PSI < 0.1: no significant shift. 0.1-0.2: moderate shift, worth watching.
>= 0.2: conventionally "significant drift".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bucket_edges(reference: pd.Series, n_buckets: int = 10) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(reference):
        quantiles = np.linspace(0, 1, n_buckets + 1)
        edges = np.unique(reference.quantile(quantiles).values)
        if len(edges) < 3:
            edges = np.array([reference.min() - 1e-9, reference.max() + 1e-9])
        return edges
    return np.array([])  # categorical handled separately


def psi_numeric(reference: pd.Series, comparison: pd.Series, n_buckets: int = 10, eps: float = 1e-4) -> float:
    edges = _bucket_edges(reference, n_buckets)
    if len(edges) < 2:
        return 0.0
    ref_counts, _ = np.histogram(reference.dropna(), bins=edges)
    cmp_counts, _ = np.histogram(comparison.dropna(), bins=edges)
    ref_pct = ref_counts / max(1, ref_counts.sum())
    cmp_pct = cmp_counts / max(1, cmp_counts.sum())
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cmp_pct = np.where(cmp_pct == 0, eps, cmp_pct)
    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))


def psi_categorical(reference: pd.Series, comparison: pd.Series, eps: float = 1e-4) -> float:
    levels = set(reference.dropna().unique()) | set(comparison.dropna().unique())
    ref_pct = reference.value_counts(normalize=True)
    cmp_pct = comparison.value_counts(normalize=True)
    total = 0.0
    for level in levels:
        r = ref_pct.get(level, eps) or eps
        c = cmp_pct.get(level, eps) or eps
        total += (c - r) * np.log(c / r)
    return float(total)


def psi_report(reference_df: pd.DataFrame, comparison_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in reference_df.columns or col not in comparison_df.columns:
            continue
        # PB-10: bool dtype (one-hot flag columns — pandas' get_dummies()
        # emits bool, not the old uint8/int8) passes
        # pd.api.types.is_numeric_dtype() == True, but calling .quantile()
        # on a boolean Series crashes inside numpy's quantile interpolation
        # with "numpy boolean subtract, the '-' operator, is not
        # supported...". Reproduced by running evaluate_oracle.py after the
        # scale-mismatch fix above (psi_reference_features.csv's one-hot
        # columns round-trip through CSV as bool dtype). A 0/1 flag is also
        # semantically a category, not a continuous quantity to bucket into
        # quantiles, so bool is excluded from the numeric branch and routed
        # to psi_categorical() instead — correct either way the dtype
        # arrives (CSV round-trip or in-memory).
        is_numeric = pd.api.types.is_numeric_dtype(reference_df[col]) and not pd.api.types.is_bool_dtype(reference_df[col])
        if is_numeric:
            score = psi_numeric(reference_df[col], comparison_df[col])
        else:
            score = psi_categorical(reference_df[col].astype(str), comparison_df[col].astype(str))
        rows.append({"feature": col, "psi": score, "significant_drift": score >= 0.2})
    return pd.DataFrame(rows)
