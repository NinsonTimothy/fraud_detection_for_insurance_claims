"""
run_all.py — B10: regenerate EVERY artifact from the final configuration, in order.

  1. clean_data                     data/cleaned/
  2. imbalance experiment (dev)     smote_vs_classweight_comparison.csv
  3. train                          selection, ablation, calibration, bands,
                                    cost grid, test evaluation, SHAP, models/
  4. evaluate_oracle                external validation + stress test
  5. sensitivity_probe              one-field-at-a-time probe
  6. generate_metrics_report        docs/CURRENT_METRICS.md
  7. generate_thesis_numbers        docs/THESIS_UPDATE_NOTES.md

Run (from backend/):  python -m app.ml.run_all
"""
from __future__ import annotations

import time


def main():
    t0 = time.time()
    from app.ml import clean_data
    clean_data.clean()
    from app.ml import model_selection_experiments
    model_selection_experiments.run_smote_vs_classweight()
    from app.ml import train
    train.main()
    from app.ml import evaluate_oracle
    evaluate_oracle.evaluate_shipped_model_on_oracle()
    evaluate_oracle.train_fresh_oracle_models()
    from app.ml.inference import FraudScoringService
    FraudScoringService._instance = None  # pick up the freshly trained artifacts
    from app.ml import sensitivity_probe
    sensitivity_probe.run()
    from app.ml import generate_metrics_report
    generate_metrics_report.main()
    from app.ml import generate_thesis_numbers
    generate_thesis_numbers.main()
    print(f"run_all finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
