"""
uncertainty.py — non-parametric bootstrap 95% confidence intervals for a
SINGLE holdout split's point-estimate metrics (SH-04).

This project already reports cross-fold uncertainty via
`train.cross_validate_model()`'s `*_mean`/`*_std` columns — that answers
"how much would this number move on a different SPLIT of the training
data." Every headline number this project reports, though, is ALSO a
single point estimate read off ONE fixed 200-row holdout split (or, for
Oracle, one fixed 15,420-row external set) — `model_comparison.csv`'s
`recall`/`roc_auc`/etc., `oracle_validation_report.json`'s
`oracle_metrics`. Those numbers had no uncertainty attached at all before
this ticket: a bare `0.735` with no sense of how much sampling noise sits
under it. Bootstrap resampling answers the complementary question: "how
much would this SAME split's number move if it were a different random
SAMPLE of the same underlying rows" — resample the (y_true, y_proba) row
pairs with replacement, recompute every metric on each resample, and take
the empirical percentile interval. Standard, model-agnostic, and doesn't
require retraining anything (unlike cross_validate_model(), which refits
per fold) — it only needs the already-computed predicted probabilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary_metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / len(y_true)
    return {"recall": recall, "precision": precision, "f1": f1, "accuracy": accuracy}


def bootstrap_metric_ci(
    y_true, proba, threshold: float | None = None, n_boot: int = 1000, ci: float = 0.95, random_state: int = 42,
) -> dict[str, dict]:
    """Bootstrap CI for roc_auc/pr_auc always; also recall/precision/f1/
    accuracy when `threshold` is given (they need a hard prediction, ROC-
    AUC/PR-AUC don't). Returns `{metric: {"ci_lower": ..., "ci_upper": ...,
    "n_boot_effective": ...}}` — the caller supplies the actual point
    estimate (already computed once, honestly, elsewhere) rather than this
    function re-deriving it from a resample mean, which would NOT be the
    same number as the real metric on the real (non-resampled) data.

    A resample that happens to contain only one class can't score ROC-AUC/
    PR-AUC (undefined) — those resamples are skipped for THOSE two metrics
    only (recall/precision/f1/accuracy are still well-defined and kept),
    which is why `n_boot_effective` can differ across metrics and can be
    below `n_boot` for small/imbalanced holdouts; it's reported so a
    reader can judge how much a given interval should be trusted."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    if n == 0:
        return {}
    rng = np.random.default_rng(random_state)
    alpha = (1 - ci) / 2

    metric_names = ["roc_auc", "pr_auc"] + (["recall", "precision", "f1", "accuracy"] if threshold is not None else [])
    boot: dict[str, list] = {m: [] for m in metric_names}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y_true[idx], proba[idx]
        has_both_classes = len(np.unique(yb)) == 2
        if has_both_classes:
            boot["roc_auc"].append(roc_auc_score(yb, pb))
            boot["pr_auc"].append(average_precision_score(yb, pb))
        if threshold is not None:
            m = _binary_metrics_at_threshold(yb, pb, threshold)
            for k in ("recall", "precision", "f1", "accuracy"):
                boot[k].append(m[k])

    result = {}
    for metric, values in boot.items():
        if not values:
            continue
        arr = np.array(values)
        result[metric] = {
            "ci_lower": float(np.percentile(arr, 100 * alpha)),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha))),
            "n_boot_effective": len(values),
        }
    return result
