# ML feature critique — is this feature set deployable, or academic-only?

> **Note on this copy:** this is the original critique of the FYP prototype's
> 94-feature model, carried into this repo verbatim as the source document
> this rebuild implements against. See `docs/REBUILD_NOTES.md` for exactly
> what changed in Aegis Risk Engine as a direct result of reading this file
> (short version: `insured_hobbies`/`insured_occupation`/`auto_make` are no
> longer fully one-hot encoded — only the specific flags this critique
> calls "real signal" are kept — bringing the feature count from 94/122 down
> to 74).

Senior-ML-engineer read of the 94 features `random_forest_final.pkl` actually uses,
grounded in the model's own saved SHAP weights (`data/processed/shap_feature_importance.csv`),
real fraud-rate numbers computed directly from `data/cleaned/insurance_claims_cleaned.csv`,
and how real insurer Special Investigations Units (SIU) actually build fraud signal —
[NICB-documented red flags](https://ethosrisk.com/blog/5-red-flags-that-should-trigger-siu-investigations/),
[ISO ClaimSearch's cross-insurer link analysis](https://www.verisk.com/company/newsroom/enhanced-iso-claimsearch-information-system-helps-insurers-process-claims-and-detect-fraud/),
and [network/link-analysis techniques](https://fraudops.ai/articles/insurance-fraud-network-link-analysis-how-its-changing-the-way-organised-fraud-gets-caught/)
used against organized fraud rings.

**Bottom line up front:** this project demonstrates the *ML pipeline* — feature
engineering discipline, model comparison, threshold tuning, SHAP explainability — very
well, and that pipeline is genuinely reusable. But roughly half the model's actual
predictive weight sits on features that would not survive contact with a real SIU
deployment, for three distinct reasons below. That's not a criticism to bury; it's
exactly the kind of finding a thesis should state plainly, and it's independently
confirmed by `generalization_and_cv_results.md`'s external validation collapsing to
random (ROC-AUC 0.496) — largely because these are the *same* features.

---

## 1. Features that are dataset artifacts, not fraud signal

### `insured_hobbies` → `is_highrisk_hobby`, `insured_hobbies_chess`, `insured_hobbies_cross-fit`

Real fraud rates computed directly from the training data:

| Hobby | Fraud rate | n | vs. 24.7% base rate |
|---|---|---|---|
| **chess** | **82.6%** | 46 | **3.3×** |
| **cross-fit** | **74.3%** | 35 | **3.0×** |
| yachting (next highest) | 30.2% | 53 | 1.2× |
| everything else | 17–30% | 47-64 each | ~1.0-1.2× |

There is no causal story connecting playing chess to committing insurance fraud. The
cliff between chess/cross-fit (74-83%) and every other hobby (17-30%) — on samples of
35-46 rows each — is the signature of a synthetic dataset where fraud was injected via a
rule that happens to correlate with these two categories, not a real behavioral pattern.
This is a widely-known quirk of this specific 1,000-row Kaggle dataset, not something
this project introduced. **A real insurer could not act on this feature** — beyond it
being statistical noise on a 35-46-row subgroup, hobby-based fraud scoring is close to
the kind of proxy discrimination regulators scrutinize insurers for (correlating a
personal lifestyle choice with risk, with no underwriting justification). `is_exec_occupation`
(36.8% vs 24.7%, n=76 — real but much milder, and occupation-based scoring carries the
same socioeconomic-proxy concern) has the identical problem at lower severity.

**Verdict: drop, or replace with something causally grounded** (see §3).

### `zip3_risk_tier_*` — leakage the project's own notebook already flags

Target-encoded from the *same 1,000 rows the model trains on* — `get_or_build_zip3_lookup()`
computes each ZIP3's fraud rate from the training data itself, not a held-out fold. The
project's own `02_preprocessing.ipynb` already calls this out as a Phase 3 leakage
caveat, and `PROTOTYPE_SETUP.md` repeats it — the honesty here is good; the fix isn't
done yet. This is the #2 feature by SHAP weight (4.85%) and its top-level column
(`zip3_risk_tier_low_risk`) is entirely unavailable outside this training set (confirmed
in the Oracle validation — no insurer external feed shares this exact 1,000-row-derived
lookup table).

**Verdict: recompute per-CV-fold at minimum; better, replace with a fraud-rate table
built from a real, larger claims history (see §3) rather than the same small sample the
model is evaluated against.**

### The 1,000-row sample size itself

Every fraud-rate number above (including the model's headline metrics) rests on ~247
fraud cases total. `generalization_and_cv_results.md`'s 5-fold CV shows PR-AUC swinging
±0.086 across folds of the *same* dataset — before ever leaving it. Most one-hot
categorical levels have 50-90 supporting rows; several ZIP3 buckets and hobby/occupation
categories have far fewer. This isn't fixable by better feature engineering — it's a
data volume problem, worth stating explicitly as a scope limitation rather than left
implicit.

---

## 2. Real signal the feature set does capture (worth keeping, stated plainly)

Not everything here is an artifact — some engineered features are legitimate and
match real SIU practice:

- `claim_to_premium_ratio`, `vehicle_claim_pct`, `injury_claim_pct` — claim size relative
  to premium/total is a standard actuarial red flag (disproportionate claims).
- `policy_age_at_incident_days` / `is_new_customer` — "new policy, big claim shortly after
  inception" is one of the best-documented real fraud indicators (matches the "unusual
  claim frequency or timing" category in NICB-aligned SIU guidance).
- **(Pre-defence update: `is_no_witness` has been REMOVED — in this dataset zero-witness claims have
  the lowest fraud rate, 20.1% vs 29.6% at two witnesses, so the flag contradicted its own premise. See
  `docs/LIMITATIONS.md`.)** Original text: `is_no_witness` / `witnesses` — zero independent witnesses is a genuine, if weak,
  signal consistent with staged-accident patterns.
- `incident_severity` (the single highest-weight feature, 17.1% of SHAP) is plausible
  in principle — severity genuinely correlates with fraud motive — but its ordinal
  encoding is inferred (`TODO-VERIFY` in `scoring.py`) rather than confirmed against the
  preprocessing notebook, and (per the generalization doc) it's also the single field
  most responsible for the external-validation collapse, because no external dataset
  encodes severity the same way. Worth keeping the *concept*, worth verifying the
  *encoding*, and worth not treating a single categorical field as irreplaceable.

---

## 3. What real SIU/insurer tooling uses that this project doesn't have

Grounded in NICB-documented red flags and ISO ClaimSearch's actual function (a real
cross-insurer claims database SIUs query specifically to catch coordinated fraud):

**Prior-claims history — the single biggest gap.** "Unusual claim frequency or prior
history" and "repeated claims with similar circumstances" are explicitly named NICB-aligned
red flags. This project's dataset is one row per claim with no policyholder-linked claim
history at all — there's no way to compute "has this person filed 3 claims this year."
This isn't a hypothetical gap: `data/raw/fraud_oracle.csv`, already sitting unused in the
repo, has exactly this field (`PastNumberOfClaims`: none / 1 / 2-4 / more than 4) and it's
one of the stronger signals in that dataset's own EDA. Adding a claims-history feature is
the highest-leverage single change available, and the data to prototype it is already here.

**Fault attribution.** Oracle's `04_pipeline_stress_test.ipynb` (this project's own
notebook) found `Fault` (Policy Holder vs. Third Party) has ~8× the fraud rate difference
between its two values — a stronger single-feature signal than almost anything in the
primary model. The primary dataset has no equivalent field at all.

**Claimant/repair-shop/attorney network-link analysis.** ISO ClaimSearch's actual
production function is cross-referencing "individuals who change roles as part of an
insurance-fraud ring" and flagging address/entity overlaps across claims — this is how
real insurers catch *organized*, multi-claim fraud rings, which single-claim scoring
structurally cannot see (an individually well-formed claim can still be part of a ring).
This project's per-row scoring architecture has no mechanism for this at all — it's an
architectural gap, not a missing column, and worth naming as future work rather than
implying the current single-claim model covers it.

**Reporting-lag and evolving-narrative signals.** NICB-documented red flags emphasize
delayed reporting paired with high severity, and claim details that change between
conversations — this project has an incident date and a policy-bind date, but no claim
*report* date distinct from the incident date, so it cannot compute reporting lag at all.

**Identity-verification irregularities** (mismatched address/DOB/contact info, synthetic
identity indicators) — not present in this schema either; a lower-priority addition
since the current dataset offers no path to it, but worth naming as a category real
tooling covers that this doesn't.

---

## Recommendation — a concrete, achievable roadmap

1. **Drop or heavily deprioritize `insured_hobbies`/`insured_occupation`-derived flags.**
   The evidence above (82.6%/74.3% fraud rates on n=35-46) doesn't support keeping them
   as production features regardless of their SHAP weight — high SHAP weight on a
   spurious correlation is a warning sign, not a validation.
2. **Recompute `zip3_risk_tier` per CV fold**, or better, source it from a larger,
   independent claims-volume table rather than the same 1,000 rows being scored.
3. **Add a claims-history feature using the Oracle dataset's `PastNumberOfClaims` as a
   prototype** — the highest-leverage single addition, and the source data already
   exists in this repo (`data/raw/fraud_oracle.csv`).
4. **Add `Fault`-equivalent attribution if the data collection process can capture it** —
   the project's own stress-test notebook already found this is a strong signal on a
   comparable dataset.
5. **State the network-link-analysis gap explicitly as future work** in the thesis —
   it's a structural limitation of any single-claim scoring architecture, not a bug, and
   naming it correctly (rather than implying the current model does something it
   doesn't) strengthens the thesis's credibility.
6. **Report the internal/external metric gap from `generalization_and_cv_results.md`
   next to these feature findings** — they're the same root cause, and showing that
   connection explicitly is a stronger contribution than either finding alone.

---

## Sources

- [5 Red Flags That Should Trigger SIU Investigations — Ethos Risk](https://ethosrisk.com/blog/5-red-flags-that-should-trigger-siu-investigations/)
- [Enhanced ISO ClaimSearch Information System — Verisk](https://www.verisk.com/company/newsroom/enhanced-iso-claimsearch-information-system-helps-insurers-process-claims-and-detect-fraud/)
- [Insurance Fraud Network Link Analysis — FraudOps](https://fraudops.ai/articles/insurance-fraud-network-link-analysis-how-its-changing-the-way-organised-fraud-gets-caught/)
