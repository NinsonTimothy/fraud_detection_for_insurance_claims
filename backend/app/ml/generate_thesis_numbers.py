"""
generate_thesis_numbers.py — E5: writes docs/THESIS_UPDATE_NOTES.md, listing
every number, table and claim in Chapters 3–5 of the thesis that changes,
as OLD -> NEW with the artifact the new value comes from.

OLD values are what the Sept 2026 thesis (Project_documentation_updated.docx)
says — historical facts, so they are constants here. NEW values are ALWAYS
read from artifacts. Run: python -m app.ml.generate_thesis_numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.reporting import roc_ci_verdict
from app.ml.risk_policy import grade_for_array

ROOT = Path(__file__).resolve().parents[3]
P, M, O = ROOT / "data" / "processed", ROOT / "models", ROOT / "data" / "external" / "oracle"


def _pct(x): return f"{x * 100:.1f}%"


def build() -> str:
    m = json.load(open(M / "metrics.json")); dec = json.load(open(P / "champion_decision.json"))
    orc = json.load(open(O / "oracle_validation_report.json")); pol = json.load(open(M / "risk_policy.json"))
    rows = {r["model"]: r for r in m["model_comparison"]}
    champ = m["primary_model"]; c = rows[champ]
    rs = pd.read_csv(P / "risk_scores_test.csv"); y = rs["y_true"].to_numpy(); s = rs["y_proba"].to_numpy()
    high = grade_for_array(s, pol) == "High"; flagged = s >= m["operating_threshold"]
    fi = pd.read_csv(P / "shap_importance_by_field.csv").set_index("source_field")["share_of_total"]
    cost = pd.read_csv(P / "cost_sensitivity.csv"); ab = pd.read_csv(P / "proxy_feature_ablation.csv").set_index("variant")
    pt = pd.read_csv(P / "pairwise_tests.csv")
    oci = orc["oracle_metrics_ci"]["roc_auc"]; verdict, _ = roc_ci_verdict(orc["oracle_metrics"]["roc_auc"], oci)

    def p_of(a, b, metric):
        r = pt[(((pt.a == a) & (pt.b == b)) | ((pt.a == b) & (pt.b == a))) & (pt.metric == metric)]
        return f"{float(r.iloc[0]['p_corrected']):.3f}" if len(r) else "—"

    T = []
    def add(sec, what, old, new, src):
        T.append(f"| {sec} | {what} | {old} | {new} | `{src}` |")

    add("3.4", "Engineered features", "70", m["n_features"], "models/metrics.json")
    add("3.3.6", "`is_no_witness`", "included", "removed (contradicts data; see LIMITATIONS)", "feature_engineering.py")
    add("3.4", "Imbalance handling", "SMOTE + class weighting", "class weighting only", "smote_vs_classweight_comparison.csv")
    add("3.5", "Selection data", "CV over all 1,000 rows (test leaked)", dec["protocol"]["data"], "champion_decision.json")
    add("3.5", "CV design", "5-fold, single run", dec["protocol"]["cv"], "champion_decision.json")
    add("3.5", "Thresholds in comparison", "RF 0.44 vs LR/XGB 0.5", dec["protocol"]["thresholds"], "champion_decision.json")
    add("3.5", "Significance test", "paired t-test (4 df)", dec["protocol"]["significance_test"], "champion_decision.json")
    add("3.5", "Selection rule", "Recall > F1 hierarchy; RF hardcoded", dec["protocol"]["selection_rule"], "champion_decision.json")
    add("3.5", "Baseline", "none", "rule: incident_severity == Major Damage", "model_selection.py")
    add("3.5/4.2", "Champion", "Random Forest", champ, "models/metrics.json")
    add("3.5", "Calibration", "none (scores called probabilities)", m["calibration"]["chosen"], "calibration_summary.csv")
    add("3.6", "Risk bands", "fixed 0.30 / 0.60", f"{pol['medium_edge']:.3f} / {pol['high_edge']:.3f} (derived)", "models/risk_policy.json")
    add("3.6", "Review threshold", "0.44", f"{m['operating_threshold']:.2f}", "models/metrics.json")
    add("3.6", "Cost-optimal threshold", "0.07 (presented as optimal)",
        f"sensitivity grid: {cost['threshold'].min():.2f}–{cost['threshold'].max():.2f} over {len(cost)} assumption sets",
        "cost_sensitivity.csv")
    add("3.6", "Low-band action", "No action — auto-approved", "Low priority — standard processing; an investigator may still review", "risk_policy.py")
    add("4.2", "RF vs LR recall p", "0.0086 (stale, hardcoded)", p_of("random_forest", "logistic_regression", "recall"), "pairwise_tests.csv")
    add("4.2", "RF vs LR F1 p", "0.0166", p_of("random_forest", "logistic_regression", "f1"), "pairwise_tests.csv")
    for met in ("pr_auc", "recall", "f1"):
        r = dec["champion_vs_rule"][met]
        add("4.2 (new)", f"Champion minus rule, {met}", "—", f"{r['mean_difference']:+.3f}, p = {r['p_corrected']:.3f}", "champion_decision.json")
    add("Table 4.1", "Champion threshold / recall / precision / F1", "RF 0.44 / 73.5% / 63.2% / 0.679",
        f"{champ} {c['threshold']:.2f} / {_pct(c['recall'])} / {_pct(c['precision'])} / {c['f1']:.3f}", "model_comparison.csv")
    add("Table 4.1", "Champion PR-AUC / ROC-AUC", "0.545 / 0.794", f"{c['pr_auc']:.3f} / {c['roc_auc']:.3f}", "model_comparison.csv")
    for n in [k for k in rows if k != champ]:
        r = rows[n]
        add("Table 4.1", f"{n}: thr / F1 / PR-AUC / ROC-AUC", "—" if n == "major_damage_rule" else "see thesis",
            f"{r['threshold']:.2f} / {r['f1']:.3f} / {r['pr_auc']:.3f} / {r['roc_auc']:.3f}", "model_comparison.csv")
    add("4.3 (new)", "Champion test Brier / ECE", "not reported", f"{c['test_brier']:.3f} / {c['test_ece']:.3f}", "model_comparison.csv")
    add("4.2.4", "Proxy ablation PR-AUC off -> on", "holdout comparison (used test set)",
        f"{ab.loc['proxy_features_off', 'pr_auc_mean']:.3f} -> {ab.loc['proxy_features_on', 'pr_auc_mean']:.3f} (dev CV)",
        "proxy_feature_ablation.csv")
    add("4.4", "Oracle ROC-AUC (95% CI)", "0.463 (0.443–0.481)",
        f"{orc['oracle_metrics']['roc_auc']:.3f} ({oci['ci_lower']:.3f}–{oci['ci_upper']:.3f})", "oracle_validation_report.json")
    add("4.4", "Oracle wording", "random", verdict, "reporting.roc_ci_verdict")
    add("4.4", "Oracle fields mapped", "7 or 10 (inconsistent)", str(orc["field_mapping_counts"]), "oracle_field_mapping.csv")
    add("4.4", "SHAP weight constant on Oracle", "91.7%", _pct(orc["share_of_shap_weight_constant_on_oracle"]), "oracle_validation_report.json")
    add("4.5", "SHAP share of incident severity", "58.8% (two columns)", _pct(fi.get("incident_severity", np.nan)), "shap_importance_by_field.csv")
    add("4.6", "High band: share of test claims / of fraud", "26.5% / 67.3%",
        f"{_pct(high.mean())} / {_pct(y[high].sum() / max(1, y.sum()))}", "risk_scores_test.csv + risk_policy.json")
    add("4.6", "Claims flagged at review threshold / recall", "57 / 73.5%",
        f"{int(flagged.sum())} / {_pct(y[flagged].sum() / max(1, y.sum()))}", "risk_scores_test.csv")
    add("4.6", "Test decisions identical to rule", "100% (not reported)",
        _pct(m["baseline_agreement"]["test_decision_agreement"]), "models/metrics.json")

    head = ["# Thesis update notes (auto-generated — do not hand-edit)", "",
            "Every number, table and claim in Chapters 3–5 that changes because of the pre-defence round-2 work. "
            "OLD = the Sept 2026 thesis. NEW = read from the named artifact by `python -m app.ml.generate_thesis_numbers`.", "",
            "| Section | What | OLD | NEW | Source artifact |", "|---|---|---|---|---|"]
    tail = ["", "## Claims to rewrite (not just numbers)", "",
            f"- **Champion.** The champion is `{champ}`, selected by a rule written in code before results "
            "(best PR-AUC unless a simpler model is not significantly worse). Remove wording that presents "
            "Random Forest as the measured winner.",
            "- **Rule baseline.** State plainly what `champion_decision.json` says under "
            "`champion_significantly_beats_rule_on` and `rule_significantly_beats_champion_on`.",
            "- **Scores** are calibrated fraud-risk scores reported with Brier and ECE; say 'fraud-risk score' throughout.",
            f"- **Oracle.** Replace 'random' with the CI-derived verdict ('{verdict}'); add the field-mapping and univariate-AUC tables.",
            "- **Witnesses.** Add the fraud-rate-by-witnesses table to Limitations as a dataset artefact.",
            "- **Cost model.** Present the cost threshold as a sensitivity analysis under stated assumptions, not an optimum.",
            "- **Leakage.** Describe the test-set leak in model selection, proxy ablation and threshold choice, and its fix.",
            "- **Decision support.** The system recommends a priority; investigators decide. No auto-approval anywhere.",
            "- **Calibration rule amendment.** Disclose that the isotonic/sigmoid preference was amended after the first run."]
    return "\n".join(head + T + tail) + "\n"


def build_viva() -> str:
    """E4: docs/VIVA_PREP.md — examiner questions with short honest answers;
    every number is read from the artifacts."""
    m = json.load(open(M / "metrics.json")); dec = json.load(open(P / "champion_decision.json"))
    orc = json.load(open(O / "oracle_validation_report.json")); pol = json.load(open(M / "risk_policy.json"))
    rows = {r["model"]: r for r in m["model_comparison"]}; champ = m["primary_model"]; c = rows[champ]
    rule = rows["major_damage_rule"]; vr = dec["champion_vs_rule"]; ba = m["baseline_agreement"]
    cal = m["calibration"]; cr = cal["results"]
    oci = orc["oracle_metrics_ci"]["roc_auc"]; verdict, long = roc_ci_verdict(orc["oracle_metrics"]["roc_auc"], oci)
    cost = pd.read_csv(P / "cost_sensitivity.csv"); ab = pd.read_csv(P / "proxy_feature_ablation.csv").set_index("variant")
    probe = pd.read_csv(P / "sensitivity_probe.csv"); sev = probe[probe.field == "incident_severity"].set_index("value")["fraud_risk_score"]
    df = pd.read_csv(ROOT / "data" / "cleaned" / "insurance_claims_cleaned.csv")
    wit = df.groupby("witnesses")["fraud_reported"].apply(lambda x: (x == "Y").mean())
    ua = pd.read_csv(O / "oracle_univariate_auc.csv")
    trail = "; ".join(f"{t['candidate']}: {t['decision']}" for t in dec["decision_trail"])
    Q = [
        ("Why this champion?",
         f"It was computed, not chosen: the rule written in code before results takes the best development PR-AUC model "
         f"unless a simpler one is not significantly worse. Best PR-AUC: {dec['best_pr_auc_model']}; trail: {trail}. "
         f"Champion: {champ} ({cal['chosen']} calibration)."),
        ("What does the model add over the severity rule?",
         f"{champion_vs_rule_sentence_safe(dec)} Development CV, champion minus rule: PR-AUC {vr['pr_auc']['mean_difference']:+.3f} "
         f"(p = {vr['pr_auc']['p_corrected']:.3f}), F1 {vr['f1']['mean_difference']:+.3f} (p = {vr['f1']['p_corrected']:.3f}). "
         f"Test: champion F1 {c['f1']:.3f} vs rule {rule['f1']:.3f}; PR-AUC {c['pr_auc']:.3f} vs {rule['pr_auc']:.3f}. "
         f"Same review decision on {_pct(ba['test_decision_agreement'])} of test claims. The model adds a ranking within "
         "severity groups, calibrated scores and per-claim explanations — not better yes/no decisions."),
        ("Why does Oracle collapse, and what does the verdict mean?",
         f"Verdict: {verdict}. {long}. Oracle lacks severity and every claim amount, so "
         f"{_pct(orc['share_of_shap_weight_constant_on_oracle'])} of the model's SHAP weight sits on features frozen at defaults. "
         f"The {len(ua)} features that still vary have development AUCs between {ua.auc_development.min():.2f} and "
         f"{ua.auc_development.max():.2f}: there was almost no transferable signal. 'Inverted' would mean the CI is entirely "
         "below 0.5 (the previous Random Forest was); 'random' means it contains 0.5."),
        ("Why do witnesses increase risk?",
         f"Because this dataset says so: fraud rate by witnesses 0/1/2/3 = {' / '.join(_pct(wit.loc[i]) for i in range(4))}. "
         "It contradicts real-world red flags, so we removed is_no_witness, kept the raw count, and disclose it as a dataset artefact."),
        ("Is the score a probability? Is it calibrated?",
         f"Development out-of-fold Brier: raw {cr['none']['brier']:.3f}, sigmoid {cr['sigmoid']['brier']:.3f}, isotonic "
         f"{cr['isotonic']['brier']:.3f}; base rate {cr['none']['brier_base_rate']:.3f}. Shipped: {cal['chosen']}. Test Brier "
         f"{c['test_brier']:.3f}, ECE {c['test_ece']:.3f}. We still call it a fraud-risk score. We amended the calibration rule "
         "after the first run (isotonic collapsed scores to a few values) and disclose that."),
        ("Why is the cost threshold only a sensitivity analysis?",
         f"It depends entirely on assumptions we cannot verify (review cost, fraudulent share, recovery rate). Across "
         f"{len(cost)} assumption sets the cost-minimising threshold ranges {cost.threshold.min():.2f}–{cost.threshold.max():.2f}, "
         f"and {int((cost.flag_rate >= 0.99).sum())} sets say 'review everything'. The review threshold "
         f"({m['operating_threshold']:.2f}) is F1-optimal on development data instead."),
        ("Why decision support, not automated decisions?",
         "The model matches a one-line rule on decisions, does not transfer to Oracle, and was trained on 1,000 US claims. "
         "It recommends a priority; an investigator decides. No band approves or denies anything."),
        ("What leakage did you find and fix?",
         "Model selection, the proxy ablation and the imbalance comparison all used CV over all 1,000 rows, so the 200 "
         "test rows influenced choices. Now every decision uses the 800 development rows; the test set is used once; "
         "a test checks the recorded row ids. Earlier: a zip-prefix feature that memorised labels (removed)."),
        ("Why not use hobby/occupation if they help?",
         f"Development PR-AUC off {ab.loc['proxy_features_off', 'pr_auc_mean']:.3f} vs on {ab.loc['proxy_features_on', 'pr_auc_mean']:.3f}. "
         "They have no causal story and risk proxy discrimination, so they are off by governance decision, with the cost shown."),
        ("Any surprising behaviour?",
         f"Severity probe: Trivial {sev.get('Trivial Damage', float('nan')):.2f}, Minor {sev.get('Minor Damage', float('nan')):.2f}, "
         f"Major {sev.get('Major Damage', float('nan')):.2f}, Total Loss {sev.get('Total Loss', float('nan')):.2f}. Total Loss "
         "scores near Minor because Total Loss claims are rarely fraud in this data. Disclosed."),
    ]
    lines = ["# Viva prep (auto-generated from artifacts — regenerate after any retrain)", "",
             "Short honest answers. Numbers come from the artifacts via `python -m app.ml.generate_thesis_numbers`.", ""]
    for q, a in Q:
        lines += [f"**Q: {q}**  ", a, ""]
    return "\n".join(lines)


def champion_vs_rule_sentence_safe(dec):
    from app.ml.reporting import champion_vs_rule_sentence
    return champion_vs_rule_sentence(dec)


def main():
    (ROOT / "docs" / "VIVA_PREP.md").write_text(build_viva())
    text = build()
    (ROOT / "docs" / "THESIS_UPDATE_NOTES.md").write_text(text)
    old = ROOT / "docs" / "THESIS_NUMBERS.md"
    if old.exists():
        old.unlink()
    print(text)


if __name__ == "__main__":
    main()
