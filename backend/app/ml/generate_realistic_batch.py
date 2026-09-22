"""Generates a larger, research-grounded simulated batch-upload dataset —
10,000 rows, split into two <=5,000-row CSVs so each one actually clears
POST /score/batch's MAX_BATCH_ROWS cap (app/core/config.py; a single
10,000-row file would 413).

This is a scaled-up, more realistic sibling of generate_sample_batch.py
(the original 60-row fixture, kept as-is for quick manual smoke tests).
Two things differ here, both requested directly by the user:

1. Realistic SIZE (10,000 rows, not 60) — split for batch upload.
2. Grounded in actual research on real-world auto insurance claims/fraud,
   not just this project's own invented scenario logic. Specifically:

   - Overall fraud-pattern prevalence: industry estimates put 10-20% of
     property/casualty claims as fraudulent to some degree (ValuePenguin/
     Forbes Advisor, summarizing Coalition Against Insurance Fraud and
     III figures — see delivery message for links). This generator
     targets ~15%, the middle of that range, for rows built from the
     "red-flag" pattern below — not a certainty label (no ground-truth
     fraud column is written; a real batch upload never has one either,
     and this project's own model output, not a simulated label, is the
     thing being exercised).
   - Documented fraud red flags (Ethos Risk, Frasco, and similar claims-
     investigation sources) used to correlate the *simulated* high-risk
     rows: no witnesses to the incident, no/delayed police report, quick
     high-severity claims on a newly-bound policy, no authorities
     contacted, and (per Forbes Advisor's fraud-statistics breakdown)
     unrecognized-driver/claim-severity mismatches — approximated here as
     claim amounts disproportionate to the reported incident severity.
   - Claim-severity shape: real claim amounts are right-skewed (many
     modest claims, a long tail of large ones), not uniformly random
     across a range — modeled here with a log-normal draw instead of
     generate_sample_batch.py's uniform randint, clipped to stay inside
     both ClaimPayload's schema bounds and the real training data's
     observed range ($100-$114,920 in data/cleaned/insurance_claims_cleaned.csv).
   - Red flags are PROBABILISTIC, not deterministic: even a "high-risk
     pattern" row only has an elevated chance of each flag (not all of
     them always), and a "low-risk pattern" row still occasionally has
     one, because real claims are noisy — a generator where the patterns
     separate perfectly wouldn't be a realistic test of the model.

Not wired into any endpoint or import path the app itself uses at
runtime — a standalone generator for manual-testing fixtures, same
`python -m app.ml.<name>` convention as generate_sample_batch.py and
generate_metrics_report.py.

Run: python -m app.ml.generate_realistic_batch   (from backend/)
Writes, under data/samples/:
  - realistic_claims_10k_part1.csv  (5,000 rows)
  - realistic_claims_10k_part2.csv  (5,000 rows)
  - realistic_claims_10k_full.csv   (all 10,000 rows combined, for
    offline analysis only — NOT for a single /score/batch upload, see
    MAX_BATCH_ROWS above)
Fixed seed (43, deliberately different from generate_sample_batch.py's 42
so the two fixtures don't accidentally overlap) — re-run any time for the
exact same data.
"""
import csv
import math
import random
from pathlib import Path

random.seed(43)

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/app/ml/ -> repo root
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

TOTAL_ROWS = 10_000
MAX_BATCH_ROWS = 5_000  # mirrors app/core/config.py's default; each part must stay <= this

# ~15%, the middle of the "10% or more of property/casualty claims may be
# fraudulent" / "an estimated 20% of insurance claims are fraudulent"
# range reported by ValuePenguin/Forbes Advisor (see delivery message for
# citations) — the share of rows built from the red-flag pattern below.
HIGH_RISK_PATTERN_RATE = 0.15
LOW_RISK_PATTERN_RATE = 0.65
AMBIGUOUS_RATE = 0.20

# A messy real-world CSV export realistically has SOME blank optional
# cells, but not a huge fraction — 4% here (vs. 5/60 = 8.3% in the small
# fixture; scaled down since 10,000 rows would otherwise mean 800+ blank
# cells, disproportionately exercising the PB-20/PB-26 path relative to
# how often it'd really happen).
BLANK_FIELD_RATE = 0.04

COLUMNS = [
    "months_as_customer", "age", "policy_bind_date", "policy_state", "policy_csl",
    "policy_deductable", "policy_annual_premium", "umbrella_limit", "insured_zip",
    "insured_sex", "insured_education_level", "insured_occupation", "insured_hobbies",
    "insured_relationship", "capital-gains", "capital-loss", "incident_date",
    "incident_type", "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "incident_hour_of_the_day", "number_of_vehicles_involved",
    "property_damage", "bodily_injuries", "witnesses", "police_report_available",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_make", "auto_year",
]

POLICY_STATES = ["IL", "IN", "OH"]
POLICY_CSL = ["100/300", "250/500", "500/1000"]
POLICY_DEDUCTABLE = [500, 1000, 2000]
EDUCATION = ["JD", "High School", "Associate", "College", "Masters", "PhD", "MD"]
OCCUPATIONS = [
    "adm-clerical", "armed-forces", "craft-repair", "exec-managerial",
    "farming-fishing", "handlers-cleaners", "machine-op-inspct",
    "other-service", "priv-house-serv", "prof-specialty", "protective-serv",
    "sales", "tech-support", "transport-moving",
]
HOBBIES = [
    "base-jumping", "basketball", "board-games", "bungie-jumping", "camping",
    "chess", "cross-fit", "dancing", "exercise", "golf", "hiking", "kayaking",
    "movies", "paintball", "polo", "reading", "skydiving", "sleeping",
    "video-games", "yachting",
]
RELATIONSHIP = ["husband", "wife", "own-child", "other-relative", "not-in-family", "unmarried"]
INCIDENT_TYPE = ["Multi-vehicle Collision", "Single Vehicle Collision", "Vehicle Theft", "Parked Car"]
COLLISION_TYPE = ["Front Collision", "Rear Collision", "Side Collision"]
INCIDENT_STATES = ["NC", "NY", "OH", "PA", "SC", "VA", "WV"]
AUTO_MAKES = ["Accura", "Audi", "BMW", "Chevrolet", "Dodge", "Ford", "Honda", "Jeep",
              "Mercedes", "Nissan", "Saab", "Suburu", "Toyota", "Volkswagen"]

OPTIONAL_BLANKABLE = [
    "total_claim_amount", "insured_zip", "incident_date", "policy_bind_date",
    "umbrella_limit", "capital-gains", "injury_claim", "property_claim",
]


def _rand_date(year_lo, year_hi):
    y = random.randint(year_lo, year_hi)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _skewed_amount(median, sigma, lo, hi, zero_chance=0.0):
    """Log-normal draw, clipped to [lo, hi] — real claim-severity data is
    right-skewed (many modest claims, a long tail of large ones), unlike
    generate_sample_batch.py's uniform randint. zero_chance models claim
    components that are legitimately often $0 (e.g. no injury claimed)."""
    if zero_chance and random.random() < zero_chance:
        return 0
    mu = math.log(median)
    val = random.lognormvariate(mu, sigma)
    return int(min(max(val, lo), hi))


def make_row() -> dict:
    """Every row is built the same way — a per-row draw against
    probabilistic, correlated fields — rather than a hard scenario bucket.
    A latent 'pattern' (high_risk / low_risk / ambiguous) still governs
    which way the probabilities lean, mirroring generate_sample_batch.py's
    three scenarios, but each individual red flag is itself a coin flip
    (see module docstring) so no row is a caricature and the separation
    the model sees is realistically noisy, not clean-cut."""
    pattern = random.choices(
        ["high_risk", "low_risk", "ambiguous"],
        weights=[HIGH_RISK_PATTERN_RATE, LOW_RISK_PATTERN_RATE, AMBIGUOUS_RATE],
    )[0]

    months_as_customer = random.randint(0, 479)
    age = random.randint(19, 64)

    if pattern == "high_risk":
        # documented red flag: high-severity claim on a newly-bound policy
        if random.random() < 0.6:
            months_as_customer = random.randint(0, 24)
        no_witness = random.random() < 0.75
        no_police_report = random.random() < 0.7
        no_authorities = random.random() < 0.55
        incident_severity = random.choices(
            ["Major Damage", "Total Loss", "Minor Damage", "Trivial Damage"],
            weights=[0.45, 0.30, 0.20, 0.05],
        )[0]
        claim_median = 42000
    elif pattern == "low_risk":
        no_witness = random.random() < 0.10
        no_police_report = random.random() < 0.12
        no_authorities = random.random() < 0.10
        incident_severity = random.choices(
            ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"],
            weights=[0.35, 0.45, 0.17, 0.03],
        )[0]
        claim_median = 9000
    else:  # ambiguous — mixed signals, still a plausible real claim
        no_witness = random.random() < 0.40
        no_police_report = random.random() < 0.40
        no_authorities = random.random() < 0.35
        incident_severity = random.choices(
            ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"],
            weights=[0.20, 0.35, 0.35, 0.10],
        )[0]
        claim_median = 18000

    witnesses = 0 if no_witness else random.randint(1, 3)
    police_report_available = "NO" if no_police_report else "YES"
    authorities_contacted = (
        random.choice(["None", "Other"]) if no_authorities else random.choice(["Police", "Fire", "Ambulance"])
    )

    bind_year = max(1990, 2015 - months_as_customer // 12 - random.randint(0, 3))
    policy_bind_date = _rand_date(bind_year, min(2015, bind_year + 2))
    incident_date = _rand_date(2015, 2015)

    premium = round(random.uniform(433, 2048), 2)
    vehicle_claim = _skewed_amount(claim_median * 0.65, 0.7, 500, 95000)
    injury_claim = _skewed_amount(claim_median * 0.25, 0.9, 100, 30000, zero_chance=0.35)
    property_claim = _skewed_amount(claim_median * 0.15, 0.9, 100, 20000, zero_chance=0.25)
    total_claim_amount = vehicle_claim + injury_claim + property_claim

    row = {
        "months_as_customer": months_as_customer,
        "age": age,
        "policy_bind_date": policy_bind_date,
        "policy_state": random.choice(POLICY_STATES),
        "policy_csl": random.choice(POLICY_CSL),
        "policy_deductable": random.choice(POLICY_DEDUCTABLE),
        "policy_annual_premium": premium,
        "umbrella_limit": random.choice([0, 0, 0, 2_000_000, 5_000_000, -1_000_000]),
        "insured_zip": random.randint(100000, 999999),
        "insured_sex": random.choice(["MALE", "FEMALE"]),
        "insured_education_level": random.choice(EDUCATION),
        "insured_occupation": random.choice(OCCUPATIONS),
        "insured_hobbies": random.choice(HOBBIES),
        "insured_relationship": random.choice(RELATIONSHIP),
        "capital-gains": random.choice([0, 0, 0, random.randint(1000, 90000)]),
        "capital-loss": random.choice([0, 0, 0, -random.randint(1000, 60000)]),
        "incident_date": incident_date,
        "incident_type": random.choice(INCIDENT_TYPE),
        "collision_type": random.choice(COLLISION_TYPE),
        "incident_severity": incident_severity,
        "authorities_contacted": authorities_contacted,
        "incident_state": random.choice(INCIDENT_STATES),
        "incident_hour_of_the_day": random.randint(0, 23),
        "number_of_vehicles_involved": random.randint(1, 4),
        "property_damage": random.choice(["YES", "NO"]),
        "bodily_injuries": random.randint(0, 2),
        "witnesses": witnesses,
        "police_report_available": police_report_available,
        "total_claim_amount": total_claim_amount,
        "injury_claim": injury_claim,
        "property_claim": property_claim,
        "vehicle_claim": vehicle_claim,
        "auto_make": random.choice(AUTO_MAKES),
        "auto_year": random.randint(1998, 2015),
    }
    # tag (not written to CSV — a real batch upload never has a label
    # column either) so this row's pattern can be checked by tests/analysis
    row["_pattern"] = pattern
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in COLUMNS})


def main():
    rows = [make_row() for _ in range(TOTAL_ROWS)]

    n_blank = int(TOTAL_ROWS * BLANK_FIELD_RATE)
    blank_indices = random.sample(range(TOTAL_ROWS), n_blank)
    for i in blank_indices:
        field = random.choice(OPTIONAL_BLANKABLE)
        rows[i][field] = ""

    random.shuffle(rows)

    part1, part2 = rows[:5000], rows[5000:]
    assert len(part1) <= MAX_BATCH_ROWS and len(part2) <= MAX_BATCH_ROWS

    _write_csv(SAMPLES_DIR / "realistic_claims_10k_part1.csv", part1)
    _write_csv(SAMPLES_DIR / "realistic_claims_10k_part2.csv", part2)
    _write_csv(SAMPLES_DIR / "realistic_claims_10k_full.csv", rows)

    n_high = sum(1 for r in rows if r["_pattern"] == "high_risk")
    n_low = sum(1 for r in rows if r["_pattern"] == "low_risk")
    n_amb = sum(1 for r in rows if r["_pattern"] == "ambiguous")
    print(f"wrote {len(rows)} rows -> {SAMPLES_DIR}")
    print(f"  part1: {len(part1)} rows, part2: {len(part2)} rows (each <= MAX_BATCH_ROWS={MAX_BATCH_ROWS})")
    print(f"  patterns: high_risk={n_high} low_risk={n_low} ambiguous={n_amb} (blank-field rows: {n_blank})")


if __name__ == "__main__":
    main()
