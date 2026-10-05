# Viva prep (auto-generated from artifacts — regenerate after any retrain)

Short honest answers. Numbers come from the artifacts via `python -m app.ml.generate_thesis_numbers`.

**Q: Why this champion?**  
It was computed, not chosen: the rule written in code before results takes the best development PR-AUC model unless a simpler one is not significantly worse. Best PR-AUC: logistic_regression; trail: logistic_regression: best PR-AUC — qualifies. Champion: logistic_regression (sigmoid calibration).

**Q: What does the model add over the severity rule?**  
No metric on which logistic_regression significantly beats the one-line major-damage rule; the rule significantly beats logistic_regression on f1. Development CV, champion minus rule: PR-AUC +0.038 (p = 0.076), F1 -0.056 (p = 0.005). Test: champion F1 0.618 vs rule 0.679; PR-AUC 0.558 vs 0.529. Same review decision on 96.0% of test claims. The model adds a ranking within severity groups, calibrated scores and per-claim explanations — not better yes/no decisions.

**Q: Why does Oracle collapse, and what does the verdict mean?**  
Verdict: indistinguishable from random. ROC-AUC 0.519, 95% CI [0.499, 0.538] contains 0.5: no measurable ranking signal. Oracle lacks severity and every claim amount, so 88.0% of the model's SHAP weight sits on features frozen at defaults. The 11 features that still vary have development AUCs between 0.48 and 0.54: there was almost no transferable signal. 'Inverted' would mean the CI is entirely below 0.5 (the previous Random Forest was); 'random' means it contains 0.5.

**Q: Why do witnesses increase risk?**  
Because this dataset says so: fraud rate by witnesses 0/1/2/3 = 20.1% / 24.4% / 29.6% / 24.7%. It contradicts real-world red flags, so we removed is_no_witness, kept the raw count, and disclose it as a dataset artefact.

**Q: Is the score a probability? Is it calibrated?**  
Development out-of-fold Brier: raw 0.195, sigmoid 0.156, isotonic 0.155; base rate 0.186. Shipped: sigmoid. Test Brier 0.139, ECE 0.076. We still call it a fraud-risk score. We amended the calibration rule after the first run (isotonic collapsed scores to a few values) and disclose that.

**Q: Why is the cost threshold only a sensitivity analysis?**  
It depends entirely on assumptions we cannot verify (review cost, fraudulent share, recovery rate). Across 36 assumption sets the cost-minimising threshold ranges 0.04–0.80, and 7 sets say 'review everything'. The review threshold (0.27) is F1-optimal on development data instead.

**Q: Why decision support, not automated decisions?**  
The model matches a one-line rule on decisions, does not transfer to Oracle, and was trained on 1,000 US claims. It recommends a priority; an investigator decides. No band approves or denies anything.

**Q: What leakage did you find and fix?**  
Model selection, the proxy ablation and the imbalance comparison all used CV over all 1,000 rows, so the 200 test rows influenced choices. Now every decision uses the 800 development rows; the test set is used once; a test checks the recorded row ids. Earlier: a zip-prefix feature that memorised labels (removed).

**Q: Why not use hobby/occupation if they help?**  
Development PR-AUC off 0.527 vs on 0.692. They have no causal story and risk proxy discrimination, so they are off by governance decision, with the cost shown.

**Q: Any surprising behaviour?**  
Severity probe: Trivial 0.12, Minor 0.15, Major 0.54, Total Loss 0.22. Total Loss scores near Minor because Total Loss claims are rarely fraud in this data. Disclosed.
