"""
cost_threshold.py — sweeps candidate probability thresholds and picks the
one that minimizes total expected cost, rather than defaulting to 0.5.

Cost model (disclosed assumption, not independently cited — same pattern
used in the sibling MoMo Guard project):
  - False negative: the claim's own `total_claim_amount` — real money paid
    out on a claim that should have been investigated.
  - False positive: a flat analyst-review-time proxy (`FP_REVIEW_COST_GHS`),
    editable — every flagged claim costs an investigator's time to clear,
    whether or not it turns out to be fraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import ANALYST_REVIEW_COST as FP_REVIEW_COST
# ^ single source of truth: config.ANALYST_REVIEW_COST reads the
# FP_REVIEW_COST env var (default 250.0, dataset's own currency units).
# Previously this module hardcoded its own 250.0 constant, so setting
# FP_REVIEW_COST had no effect on the actual cost sweep (PB-17).


def sweep_thresholds(y_true: np.ndarray, y_proba: np.ndarray, claim_amounts: np.ndarray, steps: int = 50) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, steps)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        fp_mask = (pred == 1) & (y_true == 0)
        fn_mask = (pred == 0) & (y_true == 1)
        tp_mask = (pred == 1) & (y_true == 1)
        tn_mask = (pred == 0) & (y_true == 0)
        fn_cost = claim_amounts[fn_mask].sum()
        fp_cost = fp_mask.sum() * FP_REVIEW_COST
        rows.append({
            "threshold": t, "tn": int(tn_mask.sum()), "fp": int(fp_mask.sum()),
            "fn": int(fn_mask.sum()), "tp": int(tp_mask.sum()),
            "fn_cost": float(fn_cost), "fp_cost": float(fp_cost),
            "total_cost": float(fn_cost + fp_cost),
            "recall": float(tp_mask.sum() / max(1, (y_true == 1).sum())),
            "precision": float(tp_mask.sum() / max(1, pred.sum())),
        })
    return pd.DataFrame(rows)


def find_cost_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, claim_amounts: np.ndarray, steps: int = 50) -> dict:
    sweep = sweep_thresholds(y_true, y_proba, claim_amounts, steps)
    best = sweep.loc[sweep["total_cost"].idxmin()]
    return {"threshold": float(best["threshold"]), "total_cost": float(best["total_cost"]),
            "recall": float(best["recall"]), "precision": float(best["precision"]), "sweep": sweep}
