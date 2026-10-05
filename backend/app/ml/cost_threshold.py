"""
cost_threshold.py — B6: cost-SENSITIVITY analysis (never the operating threshold).

The previous model charged a missed fraud its FULL claim amount and charged
nothing for reviewing a true positive, which mechanically pushed the
"optimal" threshold toward flagging almost everything (0.07 -> precision =
base rate). Corrected, disclosed model, at threshold t:

    expected_cost(t) = sum over missed frauds (FN) of
                           claim_amount x fraudulent_share x recovery_rate
                     + review_cost x number_flagged        (TP and FP alike)

  fraudulent_share — share of a fraudulent claim's amount that is actually
                     fraudulent (padding vs fully staged)
  recovery_rate    — share of that amount the insurer would avoid paying /
                     recover if the claim were investigated
  review_cost      — cost of one investigator review

All three are ASSUMPTIONS (config.COST_*), swept over a small grid; the
result is a table of cost-minimising thresholds, one per assumption set.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from app.core.config import COST_FRAUD_SHARES, COST_RECOVERY_RATES, COST_REVIEW_COSTS

GRID = np.round(np.linspace(0.02, 0.98, 49), 2)


def expected_cost(y, proba, amounts, t, review_cost, fraud_share, recovery_rate) -> dict:
    y, proba, amounts = np.asarray(y), np.asarray(proba), np.asarray(amounts, dtype=float)
    flag = proba >= t
    missed = (~flag) & (y == 1)
    loss = float((amounts[missed] * fraud_share * recovery_rate).sum())
    review = float(review_cost * flag.sum())
    tp = int((flag & (y == 1)).sum())
    return {"threshold": float(t), "missed_fraud_loss": loss, "review_cost_total": review,
            "total_cost": loss + review, "flag_rate": float(flag.mean()),
            "recall": tp / max(1, int((y == 1).sum())), "precision": tp / max(1, int(flag.sum()))}


def sensitivity_grid(y, proba, amounts) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (optimum per assumption set, full sweep)."""
    sweep_rows, best_rows = [], []
    for rc, fs, rr in itertools.product(COST_REVIEW_COSTS, COST_FRAUD_SHARES, COST_RECOVERY_RATES):
        rows = [{"review_cost": rc, "fraud_share": fs, "recovery_rate": rr,
                 **expected_cost(y, proba, amounts, t, rc, fs, rr)} for t in GRID]
        sweep_rows += rows
        best_rows.append(min(rows, key=lambda r: r["total_cost"]))
    return pd.DataFrame(best_rows), pd.DataFrame(sweep_rows)
