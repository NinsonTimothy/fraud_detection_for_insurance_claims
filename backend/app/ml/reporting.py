"""
reporting.py — the ONLY place result wording is decided, so the generated
docs, the dashboard and the tests can never describe the same artifact
differently. Every function takes numbers from artifacts and returns words;
none of them contains a result.
"""
from __future__ import annotations


def roc_ci_verdict(roc: float, ci: dict | None) -> tuple[str, str]:
    """(short, long) description of a ROC-AUC with its bootstrap 95% CI."""
    if not ci:
        return "no CI available", f"ROC-AUC {roc:.3f} (no confidence interval available)"
    lo, hi = ci["ci_lower"], ci["ci_upper"]
    span = f"ROC-AUC {roc:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]"
    if hi < 0.5:
        return ("significantly inverted ranking",
                f"{span} lies entirely BELOW 0.5: the model ranks genuine fraud systematically LOWER than "
                "legitimate claims on this data. This is worse than random, not 'random'")
    if lo > 0.5:
        return ("better than random, but weak",
                f"{span} lies entirely above 0.5: some ranking signal transfers, but far less than internally")
    return "indistinguishable from random", f"{span} contains 0.5: no measurable ranking signal"


def champion_vs_rule_sentence(decision: dict) -> str:
    champ = decision["champion"]
    better = decision.get("champion_significantly_beats_rule_on", [])
    worse = decision.get("rule_significantly_beats_champion_on", [])
    parts = []
    if better:
        parts.append(f"{champ} significantly beats the Major-Damage rule on {', '.join(better)}")
    else:
        parts.append(f"no metric on which {champ} significantly beats the one-line Major-Damage rule")
    if worse:
        parts.append(f"the rule significantly beats {champ} on {', '.join(worse)}")
    return "; ".join(parts).capitalize() + "."


def calibration_sentence(calib: dict) -> str:
    r = calib["results"]
    chosen = calib["chosen"]
    base = (f"Development out-of-fold Brier: raw {r['none']['brier']:.3f}, isotonic {r['isotonic']['brier']:.3f}, "
            f"sigmoid {r['sigmoid']['brier']:.3f} (predicting the base rate scores {r['none']['brier_base_rate']:.3f}).")
    return base + (f" Shipped with {chosen} calibration." if chosen != "none" else " Shipped uncalibrated.")
