"""
cost_threshold.py — sweeps candidate probability thresholds and picks the
one that minimizes total expected cost, rather than defaulting to 0.5.

Cost model (disclosed assumption, not independently cited — same pattern
used in the sibling MoMo Guard project):
  - False negative: the claim's own `total_claim_amount` — real money paid
    out on a claim that should have been investigated.
  - False positive: a flat analyst-review-time proxy (config.ANALYST_REVIEW_COST,
    set via the FP_REVIEW_COST env var), editable — every flagged claim costs
    an investigator's time to clear, whether or not it turns out to be fraud.
    (PB-21: an earlier draft of this docstring named the constant
    `FP_REVIEW_COST_GHS`, a leftover from the sibling MoMo Guard project's
    GHS-denominated cost model — this dataset's claim amounts are USD-style,
    not GHS, so the suffix was dropped; the underlying dollar figure is an
    unvalidated proxy either way, see docs/LIMITATIONS.md.)

PB-03 note: this dataset's mean `total_claim_amount` is ~$52,762 against a
default `ANALYST_REVIEW_COST` of $250 — roughly a 211x ratio. Under a pure
expected-cost objective, missing even one extra real fraud case almost
always costs more than reviewing ~211 extra false alarms, so
`find_cost_optimal_threshold()` reliably lands very close to the bottom of
the swept range (near-universal flagging) rather than some interior
tradeoff point. This is the correct, reproducible output of the disclosed
cost model, not a bug in the sweep — but it also means the result isn't a
useful OPERATING threshold on its own (an analyst team cannot review
"nearly every claim"). `train.py` reports it as a diagnostic/
sensitivity-analysis number alongside the model comparison; the actual
`operating_threshold` used to flag claims is chosen separately, by
F1-optimal search (see `train.py`'s `_f1_optimal_threshold()`).
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
