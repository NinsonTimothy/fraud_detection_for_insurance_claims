"""Generates a simulated batch-upload CSV for manually exercising
POST /score/batch (and the dashboard's Batch review page) — every raw
column engineer_features()/RAW_FEATURE_COLUMNS expects, valid values
drawn from the same ranges/vocabulary the real training data and
ClaimPayload's schema use, no ground-truth label column (a real batch
upload never has one either).

Not wired into any endpoint or import path the app itself uses at
runtime — a standalone generator for a manual-testing fixture, kept in
app/ml/ alongside generate_metrics_report.py for the same
`python -m app.ml.<name>` convention, not because scoring depends on it.

Run: python -m app.ml.generate_sample_batch   (from backend/)
Writes data/samples/sample_batch_claims.csv — re-run any time for a
fresh shuffle (the seed is fixed, so a re-run without editing this file
reproduces the exact same 60 rows).
"""
import csv
import random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/app/ml/ -> repo root, matches train.py's own convention
OUT_PATH = PROJECT_ROOT / "data" / "samples" / "sample_batch_claims.csv"

# Exact header order RAW_FEATURE_COLUMNS uses (feature_engineering.py) —
# not required by /score/batch (it reads by column NAME, not position),
# but matches the schema 1:1 so nothing here is accidentally decorative.
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


def _rand_date(year_lo, year_hi):
    y = random.randint(year_lo, year_hi)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def make_row(scenario: str, blank_fields: list[str] | None = None) -> dict:
    """scenario is one of 'low_risk', 'high_risk', 'ambiguous' — shapes the
    correlated fields that actually drive the shipped model's top SHAP
    features (is_major_damage / incident_severity_ordinal / is_no_witness /
    vehicle_claim_pct / policy_age_at_incident_days), not just noise on
    every column independently."""
    months_as_customer = random.randint(0, 479)
    age = random.randint(19, 64)
    bind_year = max(1990, 2015 - months_as_customer // 12 - random.randint(0, 3))
    policy_bind_date = _rand_date(bind_year, min(2015, bind_year + 2))
    incident_date = _rand_date(2015, 2015)

    premium = round(random.uniform(433, 2048), 2)
    vehicle_claim = random.randint(2000, 80000)
    injury_claim = random.randint(0, 20000)
    property_claim = random.randint(0, 20000)
    total_claim_amount = vehicle_claim + injury_claim + property_claim

    if scenario == "high_risk":
        incident_severity = random.choice(["Major Damage", "Total Loss"])
        witnesses = 0
        police_report_available = "NO"
        authorities_contacted = random.choice(["None", "Other"])
        # "new policy, big claim" — a real, documented red flag this
        # model actually uses (policy_age_at_incident_days).
        months_as_customer = random.randint(0, 20)
        vehicle_claim = random.randint(30000, 90000)
        total_claim_amount = vehicle_claim + injury_claim + property_claim
    elif scenario == "low_risk":
        incident_severity = random.choice(["Trivial Damage", "Minor Damage"])
        witnesses = random.randint(1, 3)
        police_report_available = "YES"
        authorities_contacted = "Police"
    else:  # ambiguous — mixed signals, still a plausible real claim
        incident_severity = random.choice(["Minor Damage", "Major Damage"])
        witnesses = random.randint(0, 2)
        police_report_available = random.choice(["YES", "NO"])
        authorities_contacted = random.choice(["Police", "Fire", "Ambulance", "Other", "None"])

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
    for f in (blank_fields or []):
        row[f] = ""
    return row


def main():
    rows = []
    for _ in range(20):
        rows.append(make_row("low_risk"))
    for _ in range(20):
        rows.append(make_row("high_risk"))
    for _ in range(15):
        rows.append(make_row("ambiguous"))

    # A handful of rows with legitimately blank OPTIONAL fields — exercises
    # apply_missing_defaults()'s documented fallback path and PB-20/PB-26's
    # NaN-handling fixes, same as a real analyst's messy CSV export would.
    optional_blankable = ["total_claim_amount", "insured_zip", "incident_date",
                           "policy_bind_date", "umbrella_limit", "capital-gains"]
    for i in range(5):
        rows.append(make_row("ambiguous", blank_fields=[optional_blankable[i]]))

    random.shuffle(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
