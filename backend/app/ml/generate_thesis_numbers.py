"""
generate_thesis_numbers.py — produces docs/THESIS_NUMBERS.md: every number
the thesis (Project_documentation_updated.docx, Sept 2026) quotes, its OLD
value, and its NEW value read from the current artifacts.

OLD values are historical facts about what the thesis currently says, so
they are listed here as constants. NEW values are always computed from
models/metrics.json, data/processed/*, data/external/oracle/* — never typed.

Run (from backend/): python -m app.ml.generate_thesis_numbers
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.risk_policy import grade_for_array

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "data" / "processed"
M = ROOT / "models"
O = ROOT / "data" / "external" / "oracle"


def _pct(x):
    return f"{x * 100:.1f}%"


def _count_tests() -> str:
    counts = []
    for d in ("backend", "dashboard"):
        try:
            out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT / d,
                                 capture_output=True, text=True, timeout=300).stdout
            line = [l for l in out.splitlines() if "collected" in l or "tests" in l][-1]
            counts.append(f"{d}: {line.strip()}")
        except Exception as e:  # pragma: no cover
            counts.append(f"{d}: could not count ({e})")
    return "; ".join(counts)


def build() -> str:
    m = json.load(open(M / "metrics.json"))
    dec = json.load(open(P / "champion_decision.json"))
    orc = json.load(open(O / "oracle_validation_report.json"))
    tests = pd.read_csv(P / "champion_pairwise_tests.csv")
    shap = pd.read_csv(P / "shap_feature_importance.csv")
    rs = pd.read_csv(P / "risk_scores_test.csv")
    champ = m["primary_model"]
    rows = {r["model"]: r for r in m["model_comparison"]}
    c = rows[champ]

    grades = grade_for_array(rs["y_proba"].to_numpy())
    y = rs["y_true"].to_numpy()
    high = grades == "High"
    flagged = rs["y_proba"].to_numpy() >= m["operating_threshold"]
    sev_share = shap.loc[shap["feature"].isin(["is_major_damage", "incident_severity_ordinal"]), "share_of_total"].sum()
    oci = orc["oracle_metrics_ci"]["roc_auc"]
    vs_rule = tests[(tests.vs == "major_damage_rule") & (tests.metric == "f1")].iloc[0]
    vs_lr_rec = tests[(tests.vs == "logistic_regression") & (tests.metric == "recall")].iloc[0]
    vs_lr_f1 = tests[(tests.vs == "logistic_regression") & (tests.metric == "f1")].iloc[0]

    L = ["# Thesis number update sheet (auto-generated — do not hand-edit)", "",
         "OLD = what `Project_documentation_updated.docx` currently says. NEW = read from the artifacts on disk "
         "when this file was generated (`python -m app.ml.generate_thesis_numbers`). Section numbers follow the "
         "Sept 2026 rewrite.", "",
         "| Thesis location | What | OLD | NEW |", "|---|---|---|---|"]
    add = lambda loc, what, old, new: L.append(f"| {loc} | {what} | {old} | {new} |")
    add("3.4.1 / Abstract", "Engineered feature count", "70", str(m["n_features"]))
    add("3.3.6", "`is_no_witness` feature", "included", "removed (dataset artefact — see LIMITATIONS)")
    add("3.5", "Champion selection data", "paired 5-fold CV on all 1,000 rows", dec["protocol"]["data"])
    add("3.5", "Selection CV design", "5 folds, fixed 0.5 threshold", f"{dec['protocol']['outer_cv']}; {dec['protocol']['inner_cv']}")
    add("3.5", "Baseline", "none", "one-line rule: flag if incident_severity = Major Damage")
    add("3.5 / 4.2", "Significance test", "paired t-test, 4 df", dec["protocol"]["significance_test"])
    add("4.2", "RF vs LR recall p-value", "0.0086 (stale, hardcoded)", f"{vs_lr_rec['p_corrected']:.3f} (not significant)")
    add("4.2", "RF vs LR F1 p-value", "0.0166", f"{vs_lr_f1['p_corrected']:.3f} ({'significant' if vs_lr_f1['significant_at_0_05'] else 'not significant'})")
    add("4.2 (new)", "Champion vs rule, F1 difference / p", "—", f"{vs_rule['mean_difference']:+.3f} / {vs_rule['p_corrected']:.3f}")
    add("4.2 (new)", "Nested-CV F1: champion vs rule", "—", f"{dec['champion_cv']['f1_mean']:.3f} vs {dec['baseline_cv']['f1_mean']:.3f}")
    add("4.2 (new)", "Test decisions identical to rule", "—", _pct(m["baseline_agreement"]["test_decision_agreement"]))
    add("Table 4.1", f"Champion", "Random Forest", champ)
    add("Table 4.1", "Operating threshold", "0.44", f"{m['operating_threshold']:.2f}")
    add("Table 4.1", "Champion F1 / Recall / Precision", "0.679 / 73.5% / 63.2%", f"{c['f1']:.3f} / {_pct(c['recall'])} / {_pct(c['precision'])}")
    add("Table 4.1", "Champion ROC-AUC / PR-AUC", "0.794 / 0.545", f"{c['roc_auc']:.3f} / {c['pr_auc']:.3f}")
    for name in ("logistic_regression", "xgboost", "major_damage_rule"):
        r = rows[name]
        add("Table 4.1", f"{name}: thr / F1 / Recall / ROC-AUC", "see old table" if name != "major_damage_rule" else "— (new row)",
            f"{r['threshold']:.2f} / {r['f1']:.3f} / {_pct(r['recall'])} / {r['roc_auc']:.3f}")
    add("4.4", "Oracle ROC-AUC (95% CI)", "0.463 (0.443–0.481)", f"{orc['oracle_metrics']['roc_auc']:.3f} ({oci['ci_lower']:.3f}–{oci['ci_upper']:.3f})")
    add("4.4", "Oracle wording", "random / indistinguishable from random", orc.get("roc_auc_verdict", "").replace("_", " "))
    add("4.4", "SHAP weight constant on Oracle", "91.7%", _pct(orc["share_of_shap_weight_constant_on_oracle"]))
    add("4.5", "SHAP share: is_major_damage + incident_severity_ordinal", "58.8%", _pct(sev_share))
    add("4.6", "High band: share of test claims", "26.5%", _pct(high.mean()))
    add("4.6", "High band: share of fraud captured", "67.3%", _pct(y[high].sum() / max(1, y.sum())))
    add("4.6", "Claims flagged at operating threshold / recall", "57 / 73.5%", f"{int(flagged.sum())} / {_pct(y[flagged].sum() / max(1, y.sum()))}")
    add("3.6", "Low band action wording", "auto-approved", "recommend standard claims handling — handler decides")
    add("3.8", "Test count", "153", _count_tests())
    L += ["", "Text changes that go with the numbers:", "",
          "- Ch. 4/5: state plainly that no ML model beat the Major-Damage rule on F1; the model's contribution is ranking within groups and explanation.",
          "- Ch. 4.4 / 5: replace every 'random' description of Oracle with 'significantly inverted (95% CI entirely below 0.5)'.",
          "- Ch. 5 Limitations: add the witness artefact paragraph from docs/LIMITATIONS.md.",
          "- Ch. 3.6/3.7: decision-support wording — the system recommends, the investigator decides.",
          "- Ch. 3.5: describe the earlier test-set leak in model selection and how it was fixed.", ""]
    return "\n".join(L)


def main():
    text = build()
    (ROOT / "docs" / "THESIS_NUMBERS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
