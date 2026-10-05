"""
form_spec.py — single source of truth for what the Score-a-claim form
collects (A5) and what it deliberately does not (D1 coverage test).

Every raw field the feature pipeline reads must be either COLLECTED on the
form, DERIVED from collected fields, or listed in DOCUMENTED_DEFAULTED with
a reason. backend/tests/test_round2.py asserts that, and that the defaulted
set carries < 5% of the model's global SHAP importance.
"""
from __future__ import annotations

from app.core.config import INCLUDE_PROXY_FEATURES

FORM_FIELDS = {
    "months_as_customer", "age", "policy_bind_date", "policy_state", "policy_csl", "policy_deductable",
    "policy_annual_premium", "umbrella_limit", "insured_sex", "insured_education_level", "insured_relationship",
    "capital-gains", "capital-loss", "incident_date", "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "incident_state", "incident_hour_of_the_day", "number_of_vehicles_involved",
    "property_damage", "bodily_injuries", "witnesses", "police_report_available",
    "injury_claim", "property_claim", "vehicle_claim", "auto_year",
} | ({"insured_hobbies", "insured_occupation"} if INCLUDE_PROXY_FEATURES else set())

DERIVED_ON_FORM = {"total_claim_amount": "injury_claim + property_claim + vehicle_claim (read-only)"}

DOCUMENTED_DEFAULTED = {
    "insured_zip": "removed from the model (PB-02); collected nowhere",
    "auto_make": "not a model feature",
    **({} if INCLUDE_PROXY_FEATURES else {
        "insured_hobbies": "proxy feature, OFF by default (governance)",
        "insured_occupation": "proxy feature, OFF by default (governance)",
    }),
}
