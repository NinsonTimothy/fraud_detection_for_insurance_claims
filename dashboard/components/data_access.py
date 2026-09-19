"""components/data_access.py — loads artifacts + provides the in-process
FraudScoringService the whole dashboard shares (same instance the API
would use, so the two surfaces can never disagree about a score)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_ROOT.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_ORACLE_DIR = PROJECT_ROOT / "data" / "external" / "oracle"


@st.cache_resource
def get_scoring_service():
    from app.ml.inference import FraudScoringService
    return FraudScoringService.instance()


def models_are_available() -> bool:
    return (MODELS_DIR / "random_forest_final.pkl").exists()


@st.cache_data
def load_metrics() -> dict:
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "model_comparison.csv")


@st.cache_data
def load_cross_validation() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cross_validation_results.csv")


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "shap_feature_importance.csv")


@st.cache_data
def load_cost_sweep() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cost_threshold_sweep.csv")


def oracle_results_available() -> bool:
    return (EXTERNAL_ORACLE_DIR / "oracle_validation_report.json").exists()


@st.cache_data
def load_oracle_report() -> dict:
    with open(EXTERNAL_ORACLE_DIR / "oracle_validation_report.json") as f:
        return json.load(f)


@st.cache_data
def load_oracle_psi() -> pd.DataFrame:
    path = EXTERNAL_ORACLE_DIR / "oracle_psi_report.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_oracle_model_comparison() -> pd.DataFrame:
    path = PROCESSED_DIR / "oracle_model_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
