"""
ML Training Pipeline
====================
Models:
  1. Cost Estimator      — GradientBoostingRegressor  → cost_reg.pkl
  2. Effort Estimator    — GradientBoostingRegressor  → effort_reg.pkl
  3. SDLC Classifier     — RandomForestClassifier     → sdlc_clf.pkl

Artifacts saved:
  scaler.pkl, cost_reg.pkl, effort_reg.pkl, sdlc_clf.pkl, metrics.pkl
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sdlc_dataset.csv")

FEATURE_COLS = [
    "team_size", "duration_months", "budget_usd",
    "requirements_clarity", "client_involvement", "tech_complexity",
    "risk_encoded", "type_encoded", "budget_per_person", "complexity_risk",
]

SDLC_COST_FACTOR    = {"Waterfall": 1.00, "Agile": 1.15, "Scrum": 1.20,
                        "Kanban": 0.85, "Spiral": 1.45, "Iterative": 1.10}
TYPE_COMPLEXITY_BOOST = {1: 0.00, 2: 0.10, 3: 0.30, 4: 0.20}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _label_sdlc(clarity, involvement, complexity, risk, team_size, duration):
    if team_size <= 6 and duration <= 6 and complexity <= 3:
        return "Kanban"
    if risk == 3 and complexity >= 4:
        return "Spiral"
    if risk == 3 and complexity == 3 and duration >= 18:
        return "Spiral"
    if clarity >= 4 and risk == 1:
        return "Waterfall"
    if clarity == 5 and risk <= 2 and involvement <= 2:
        return "Waterfall"
    if involvement >= 4 and clarity <= 3:
        return "Scrum"
    if involvement >= 4 and complexity >= 3 and risk <= 2:
        return "Scrum"
    if clarity <= 2 and involvement <= 2:
        return "Iterative"
    if clarity == 1 and risk >= 2:
        return "Iterative"
    return "Agile"


def generate_dataset(n: int = 5000) -> pd.DataFrame:
    rng  = np.random.default_rng(42)
    rows = []

    for _ in range(n):
        team_size            = int(rng.integers(2, 51))
        duration_months      = int(rng.integers(1, 49))
        budget_usd           = round(float(rng.uniform(10_000, 1_000_000)), 2)
        requirements_clarity = int(rng.integers(1, 6))
        client_involvement   = int(rng.integers(1, 6))
        tech_complexity      = int(rng.integers(1, 6))
        risk_encoded         = int(rng.integers(1, 4))
        type_encoded         = int(rng.integers(1, 5))

        budget_per_person = round(budget_usd / max(team_size, 1), 4)
        complexity_risk   = tech_complexity * risk_encoded

        sdlc = _label_sdlc(requirements_clarity, client_involvement,
                            tech_complexity, risk_encoded, team_size, duration_months)

        utilisation = 0.55 + (tech_complexity - 1) * 0.05
        effort_base = team_size * duration_months * utilisation
        effort = effort_base * (1 + (risk_encoded - 1) * 0.15) * (1 + TYPE_COMPLEXITY_BOOST[type_encoded])
        effort = round(float(effort + rng.normal(0, effort * 0.05)), 2)
        effort = max(effort, 1.0)

        base_rate = 5_000 + (tech_complexity - 1) * 2_500
        cost_base = effort * base_rate * SDLC_COST_FACTOR[sdlc]
        cost_base = min(cost_base, budget_usd * 1.2)
        cost      = round(float(max(cost_base + rng.normal(0, cost_base * 0.08), 5_000)), 2)

        rows.append([team_size, duration_months, budget_usd,
                     requirements_clarity, client_involvement, tech_complexity,
                     risk_encoded, type_encoded, budget_per_person, complexity_risk,
                     sdlc, effort, cost])

    return pd.DataFrame(rows, columns=FEATURE_COLS + ["sdlc", "effort_person_months", "cost_usd"])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame):
    """
    Returns scaled feature matrix, encoded SDLC labels,
    effort targets, cost targets, fitted scaler and label encoder.
    """
    X = df[FEATURE_COLS].copy()

    # Clip outliers at 1st / 99th percentile for numeric stability
    for col in ["budget_usd", "budget_per_person"]:
        lo, hi = X[col].quantile([0.01, 0.99])
        X.loc[:, col] = X[col].clip(lo, hi)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_clf    = le.fit_transform(df["sdlc"].values)
    y_effort = df["effort_person_months"].values.astype(float)
    y_cost   = df["cost_usd"].values.astype(float)

    return X_scaled, y_clf, y_effort, y_cost, scaler, le


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n── {label} ──")
    print(f"  MAE  : {mae:>12,.2f}")
    print(f"  RMSE : {rmse:>12,.2f}")
    print(f"  R²   : {r2:>12.4f}")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 classes: list, label: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    print(f"\n── {label} ──")
    print(f"  Accuracy : {acc:.2%}")
    print(f"  F1 (wtd) : {f1:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=classes))
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_cost_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )


def _build_effort_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    )


def _build_clf_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    df = generate_dataset(5000)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset : {len(df):,} rows  |  {df['sdlc'].value_counts().to_dict()}")

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    X, y_clf, y_effort, y_cost, scaler, le = preprocess(df)

    (X_train, X_test,
     yc_train, yc_test,
     ye_train, ye_test,
     yr_train, yr_test) = train_test_split(
        X, y_clf, y_effort, y_cost,
        test_size=0.2, random_state=42, stratify=y_clf,
    )

    print(f"\nTrain : {len(X_train):,}  |  Test : {len(X_test):,}")

    # ── 3. Cost regression ───────────────────────────────────────────────────
    cost_model = _build_cost_model()
    cost_model.fit(X_train, yr_train)
    cost_metrics = _reg_metrics(yr_test, cost_model.predict(X_test), "Cost Estimator (USD)")

    # ── 4. Effort regression ─────────────────────────────────────────────────
    effort_model = _build_effort_model()
    effort_model.fit(X_train, ye_train)
    effort_metrics = _reg_metrics(ye_test, effort_model.predict(X_test), "Effort Estimator (person-months)")

    # ── 5. SDLC classification ───────────────────────────────────────────────
    clf_model = _build_clf_model()
    clf_model.fit(X_train, yc_train)
    clf_metrics = _clf_metrics(
        yc_test, clf_model.predict(X_test),
        list(le.classes_), "SDLC Classifier"
    )

    # Cross-validation sanity check (single-process, safe on all Python versions)
    cv_scores = cross_val_score(clf_model, X, y_clf, cv=5, scoring="accuracy", n_jobs=1)
    print(f"  CV Accuracy : {cv_scores.mean():.2%} ± {cv_scores.std():.4f}")
    clf_metrics["cv_accuracy_mean"] = round(cv_scores.mean(), 4)
    clf_metrics["cv_accuracy_std"]  = round(cv_scores.std(), 4)

    # Feature importance
    fi = pd.Series(clf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances (SDLC classifier):")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 60)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    # ── 6. Save artifacts ────────────────────────────────────────────────────
    artifacts = {
        "scaler.pkl":     scaler,
        "cost_reg.pkl":   cost_model,
        "effort_reg.pkl": effort_model,
        "sdlc_clf.pkl":   clf_model,
    }
    for fname, obj in artifacts.items():
        path = os.path.join(MODEL_DIR, fname)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    # Label encoder saved separately (needed for inference)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    metrics = {
        "cost_estimator":   cost_metrics,
        "effort_estimator": effort_metrics,
        "sdlc_classifier":  clf_metrics,
        "feature_importance": fi.round(4).to_dict(),
        "sdlc_distribution":  df["sdlc"].value_counts().to_dict(),
        "dataset_size":       len(df),
    }
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    print("\n✓ All artifacts saved to", MODEL_DIR)
    return metrics


if __name__ == "__main__":
    train()
