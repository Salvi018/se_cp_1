"""
Standalone Model Evaluation
============================
Loads saved .pkl models and runs a full evaluation report
against the dataset without retraining.

Usage:
    python app/ml/evaluate.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sdlc_dataset.csv")

FEATURE_COLS = [
    "team_size", "duration_months", "budget_usd",
    "requirements_clarity", "client_involvement", "tech_complexity",
    "risk_encoded", "type_encoded", "budget_per_person", "complexity_risk",
    "team_experience", "regulatory_compliance", "geographic_distribution",
]

SEP = "=" * 60


def _load(fname: str):
    with open(os.path.join(MODEL_DIR, fname), "rb") as f:
        return pickle.load(f)


def _print_reg_report(y_true: np.ndarray, y_pred: np.ndarray, label: str, unit: str = ""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  MAE   : {mae:>14,.2f} {unit}")
    print(f"  RMSE  : {rmse:>14,.2f} {unit}")
    print(f"  MAPE  : {mape:>13.2f} %")
    print(f"  R²    : {r2:>14.4f}")

    residuals = y_true - y_pred
    print(f"\n  Residual stats:")
    print(f"    Mean  : {residuals.mean():>12,.2f}")
    print(f"    Std   : {residuals.std():>12,.2f}")
    print(f"    Min   : {residuals.min():>12,.2f}")
    print(f"    Max   : {residuals.max():>12,.2f}")

    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "mape": round(mape, 2), "r2": round(r2, 4)}


def _print_clf_report(y_true: np.ndarray, y_pred: np.ndarray,
                      y_proba: np.ndarray, classes: list):
    acc       = accuracy_score(y_true, y_pred)
    f1_wtd    = f1_score(y_true, y_pred, average="weighted")
    f1_macro  = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall    = recall_score(y_true, y_pred, average="weighted")

    print(f"\n{SEP}")
    print("  SDLC Classifier")
    print(SEP)
    print(f"  Accuracy          : {acc:.4f}  ({acc:.2%})")
    print(f"  F1 (weighted)     : {f1_wtd:.4f}")
    print(f"  F1 (macro)        : {f1_macro:.4f}")
    print(f"  Precision (wtd)   : {precision:.4f}")
    print(f"  Recall (wtd)      : {recall:.4f}")

    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    print("  Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    header = "        " + "  ".join(f"{c[:6]:>6}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i][:8]:<8}" + "  ".join(f"{v:>6}" for v in row))

    # Per-class confidence (mean max-proba)
    print("\n  Mean prediction confidence per class:")
    for i, cls in enumerate(classes):
        mask = y_pred == i
        if mask.sum() > 0:
            mean_conf = y_proba[mask].max(axis=1).mean()
            print(f"    {cls:<12} : {mean_conf:.2%}")

    return {
        "accuracy": round(acc, 4), "f1_weighted": round(f1_wtd, 4),
        "f1_macro": round(f1_macro, 4), "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def evaluate():
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Run train.py first.")
        sys.exit(1)

    # ── Load artifacts ───────────────────────────────────────────────────────
    scaler       = _load("scaler.pkl")
    cost_model   = _load("cost_reg.pkl")
    effort_model = _load("effort_reg.pkl")
    clf_model    = _load("sdlc_clf.pkl")
    le           = _load("label_encoder.pkl")

    # ── Load & preprocess data ───────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    for col in ["budget_usd", "budget_per_person"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df.loc[:, col] = df[col].clip(lo, hi)

    X        = scaler.transform(df[FEATURE_COLS].values)
    y_clf    = le.transform(df["sdlc"].values)
    y_effort = df["effort_person_months"].values.astype(float)
    y_cost   = df["cost_usd"].values.astype(float)

    _, X_test, _, yc_test, _, ye_test, _, yr_test = train_test_split(
        X, y_clf, y_effort, y_cost,
        test_size=0.2, random_state=42, stratify=y_clf,
    )

    print(f"\n{'=' * 60}")
    print("  MODEL EVALUATION REPORT")
    print(f"  Test set size : {len(X_test):,} samples")
    print(f"{'=' * 60}")

    # ── Regression reports ───────────────────────────────────────────────────
    _print_reg_report(yr_test, cost_model.predict(X_test),
                      "Cost Estimator", unit="USD")
    _print_reg_report(ye_test, effort_model.predict(X_test),
                      "Effort Estimator", unit="person-months")

    # ── Classification report ────────────────────────────────────────────────
    y_pred  = clf_model.predict(X_test)
    y_proba = clf_model.predict_proba(X_test)
    _print_clf_report(yc_test, y_pred, y_proba, list(le.classes_))

    # ── Feature importance ───────────────────────────────────────────────────
    fi = pd.Series(clf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n{SEP}")
    print("  Feature Importances (SDLC Classifier)")
    print(SEP)
    for feat, imp in fi.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    evaluate()
