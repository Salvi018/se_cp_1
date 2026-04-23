#!/usr/bin/env bash
set -e

echo "==> Installing dependencies"
pip install -r requirements.txt

echo "==> Checking ML models"
MODEL_DIR="app/ml/models"
if [ ! -f "$MODEL_DIR/sdlc_clf.pkl" ]; then
  echo "==> Training ML models (first deploy)"
  python3 app/ml/train.py
else
  echo "==> Models already exist, skipping training"
fi

echo "==> Build complete"
