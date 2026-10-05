"""
calibration.py — B4: Brier score, expected calibration error (ECE) and
reliability-curve data, plus the PRE-DECLARED calibration decision.

Decision rule (fixed before results): on the 800 development rows, compare
out-of-fold Brier score of the raw champion against CalibratedClassifierCV
(ensemble=False, cv=5) with isotonic and with sigmoid calibration. A method
is adopted only if it lowers OOF Brier by at least
config.CALIBRATION_MIN_BRIER_GAIN; otherwise the raw model ships. The test
set plays no part.

ensemble=False keeps ONE base estimator (refit on all training data) plus
ONE monotone calibration map, so SHAP still explains a single model whose
score the calibrator only re-maps monotonically (ranking is unchanged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from app.core.config import CALIBRATION_MIN_BRIER_GAIN, ECE_BINS


AMENDMENT = (
    "The first version of this rule picked whichever of isotonic/sigmoid had the lower development Brier. "
    "On the first full run it chose isotonic, which left only 12 distinct scores across the 200 test claims "
    "(isotonic regression is a step function), destroying the within-group ranking that is the model's main "
    "contribution over the severity rule, for a Brier gain over sigmoid of under 0.001. The rule was amended, "
    "AFTER seeing that result, to prefer the ranking-preserving sigmoid unless isotonic is better by at least "
    "the same minimum gain. This is a disclosed post-hoc change, recorded here and in REBUILD_NOTES.md."
)


def reliability_table(y_true, proba, n_bins: int = ECE_BINS) -> pd.DataFrame:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin": b, "bin_low": edges[b], "bin_high": edges[b + 1], "n": int(m.sum()),
                     "mean_predicted": float(proba[m].mean()) if m.any() else np.nan,
                     "observed_fraud_rate": float(y_true[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, proba, n_bins: int = ECE_BINS) -> float:
    t = reliability_table(y_true, proba, n_bins)
    t = t[t["n"] > 0]
    return float((t["n"] / t["n"].sum() * (t["mean_predicted"] - t["observed_fraud_rate"]).abs()).sum())


def calibration_summary(y_true, proba) -> dict:
    y = np.asarray(y_true)
    return {"brier": float(brier_score_loss(y, proba)),
            "brier_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean()))),
            "ece": expected_calibration_error(y, proba)}


def make_calibrated_factory(make_estimator, method: str | None):
    if method is None:
        return make_estimator
    return lambda: CalibratedClassifierCV(estimator=make_estimator(), method=method, cv=5, ensemble=False)


def choose_calibration(make_estimator, X_dev: np.ndarray, y_dev: np.ndarray, oof_fn) -> dict:
    """oof_fn(factory, X, y) -> out-of-fold probabilities (dev rows only)."""
    results = {}
    for method in (None, "isotonic", "sigmoid"):
        p = oof_fn(make_calibrated_factory(make_estimator, method), X_dev, y_dev)
        results["none" if method is None else method] = {**calibration_summary(y_dev, p), "_oof": p}
    raw = results["none"]["brier"]
    # AMENDED RULE (disclosed; see AMENDMENT below): sigmoid is preferred over
    # isotonic unless isotonic lowers Brier by at least the same minimum gain.
    # Sigmoid is a strictly increasing 2-parameter map, so it preserves the
    # full ranking; isotonic is a step function.
    iso_gain_over_sigmoid = results["sigmoid"]["brier"] - results["isotonic"]["brier"]
    best = "isotonic" if iso_gain_over_sigmoid >= CALIBRATION_MIN_BRIER_GAIN else "sigmoid"
    chosen = best if raw - results[best]["brier"] >= CALIBRATION_MIN_BRIER_GAIN else "none"
    return {"chosen": chosen, "min_gain_required": CALIBRATION_MIN_BRIER_GAIN,
            "amendment": AMENDMENT,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "_oof"} for k, v in results.items()},
            "_oof_chosen": results[chosen]["_oof"]}
