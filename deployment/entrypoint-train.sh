#!/bin/sh
# entrypoint-train.sh — one-shot init container: cleans the raw data,
# trains every model, runs the Oracle external-validation report, so
# `docker compose up --build` produces a fully working demo from a clean
# checkout with zero manual steps (same pattern as the sibling MoMo Guard
# project's entrypoint.sh).
set -e
cd /srv/backend
if [ ! -f /srv/models/random_forest_final.pkl ]; then
  echo "No trained model found — running full pipeline..."
  python -m app.ml.clean_data
  python -m app.ml.train
  python -m app.ml.evaluate_oracle
else
  echo "Trained model artifacts already present, skipping pipeline."
fi
