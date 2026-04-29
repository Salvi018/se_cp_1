"""
ML Training Pipeline v3
========================
SDLC Labels (10):
  Waterfall, Agile, Scrum, Kanban, Spiral, Iterative,
  RAD, XP, SAFe, V-Model
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sdlc_dataset.csv")

FEATURE_COLS = [
    "team_size", "duration_months", "budget_usd",
    "requirements_clarity", "client_involvement", "tech_complexity",
    "risk_encoded", "type_encoded", "budget_per_person", "complexity_risk",
    "team_experience", "regulatory_compliance", "geographic_distribution",
]

SDLC_COST_FACTOR = {
    "Waterfall": 1.00,
    "Agile":     1.15,
    "Scrum":     1.20,
    "Kanban":    0.85,
    "Spiral":    1.45,
    "Iterative": 1.10,
    "RAD":       1.05,
    "XP":        1.25,
    "SAFe":      1.35,
    "V-Model":   1.30,
}

TYPE_COMPLEXITY_BOOST    = {1: 0.00, 2: 0.10, 3: 0.30, 4: 0.20}
EXPERIENCE_EFFORT_FACTOR = {1: 1.30, 2: 1.15, 3: 1.00, 4: 0.88, 5: 0.75}
COMPLIANCE_COST_FACTOR   = {0: 1.00, 1: 1.25}
GEO_EFFORT_FACTOR        = {1: 1.00, 2: 1.12, 3: 1.25}

# ---------------------------------------------------------------------------
# Real-world seed rows
# ---------------------------------------------------------------------------
REAL_WORLD_SEEDS = [
    # (team, dur, budget, clarity, involv, complex, risk, type, exp, comply, geo, sdlc)

    # Waterfall
    (25, 36, 800000,  5, 1, 3, 1, 3, 3, 1, 1, "Waterfall"),
    (40, 48, 950000,  5, 2, 4, 1, 3, 4, 1, 1, "Waterfall"),
    (15, 24, 300000,  4, 1, 2, 1, 1, 3, 0, 1, "Waterfall"),
    (30, 36, 600000,  5, 1, 3, 1, 2, 4, 1, 1, "Waterfall"),
    (50, 60, 1200000, 5, 1, 4, 1, 3, 5, 1, 1, "Waterfall"),

    # Agile
    (8,  12, 150000,  3, 3, 3, 2, 1, 3, 0, 2, "Agile"),
    (12, 18, 220000,  3, 3, 4, 2, 4, 3, 0, 2, "Agile"),
    (10, 24, 300000,  4, 4, 3, 2, 1, 4, 0, 1, "Agile"),
    (15, 12, 180000,  3, 3, 3, 2, 2, 3, 0, 2, "Agile"),
    (20, 24, 400000,  4, 3, 4, 2, 4, 4, 0, 2, "Agile"),

    # Scrum
    (6,  6,  80000,   2, 5, 3, 2, 1, 3, 0, 1, "Scrum"),
    (8,  9,  120000,  2, 4, 4, 2, 2, 3, 0, 2, "Scrum"),
    (10, 12, 160000,  3, 5, 3, 2, 1, 4, 0, 1, "Scrum"),
    (7,  12, 100000,  2, 5, 3, 2, 1, 3, 0, 3, "Scrum"),
    (12, 9,  140000,  3, 4, 4, 2, 4, 3, 0, 2, "Scrum"),

    # Kanban
    (3,  3,  20000,   3, 4, 2, 1, 1, 3, 0, 1, "Kanban"),
    (4,  4,  30000,   4, 3, 1, 1, 1, 4, 0, 1, "Kanban"),
    (2,  6,  25000,   3, 5, 2, 1, 2, 3, 0, 1, "Kanban"),
    (5,  5,  40000,   4, 4, 2, 1, 1, 4, 0, 1, "Kanban"),
    (3,  4,  22000,   3, 3, 1, 1, 1, 3, 0, 1, "Kanban"),

    # Spiral
    (20, 30, 500000,  3, 2, 5, 3, 3, 4, 1, 1, "Spiral"),
    (15, 24, 350000,  2, 2, 5, 3, 3, 4, 1, 1, "Spiral"),
    (25, 36, 700000,  3, 1, 4, 3, 3, 5, 1, 1, "Spiral"),
    (10, 18, 280000,  2, 2, 5, 3, 4, 4, 1, 2, "Spiral"),
    (30, 48, 900000,  3, 1, 5, 3, 3, 5, 1, 1, "Spiral"),

    # Iterative
    (8,  20, 120000,  1, 2, 3, 2, 4, 3, 0, 2, "Iterative"),
    (12, 24, 200000,  2, 1, 4, 2, 4, 3, 0, 3, "Iterative"),
    (6,  18, 90000,   1, 1, 3, 2, 1, 2, 0, 2, "Iterative"),
    (10, 30, 180000,  2, 2, 3, 3, 4, 3, 0, 2, "Iterative"),
    (15, 24, 250000,  1, 2, 4, 2, 4, 4, 0, 3, "Iterative"),

    # RAD — short, prototype-driven, high involvement
    (6,  4,  60000,   2, 4, 3, 1, 1, 3, 0, 1, "RAD"),
    (8,  3,  80000,   3, 5, 2, 1, 2, 4, 0, 1, "RAD"),
    (5,  5,  50000,   2, 4, 2, 1, 1, 3, 0, 1, "RAD"),
    (10, 4,  90000,   3, 4, 3, 1, 2, 4, 0, 2, "RAD"),
    (7,  3,  70000,   2, 5, 2, 1, 1, 4, 0, 1, "RAD"),

    # XP — small expert team, TDD, pair programming
    (4,  12, 100000,  2, 5, 4, 2, 1, 5, 0, 1, "XP"),
    (5,  9,  90000,   2, 5, 4, 2, 4, 5, 0, 1, "XP"),
    (3,  12, 80000,   3, 5, 5, 2, 1, 5, 0, 1, "XP"),
    (6,  6,  110000,  2, 4, 4, 2, 2, 5, 0, 2, "XP"),
    (4,  9,  95000,   2, 5, 3, 1, 1, 5, 0, 1, "XP"),

    # SAFe — large enterprise, multiple teams
    (50, 24, 900000,  3, 3, 4, 2, 1, 3, 0, 3, "SAFe"),
    (80, 36, 1500000, 3, 3, 4, 2, 4, 4, 0, 3, "SAFe"),
    (60, 24, 1100000, 4, 3, 3, 2, 1, 4, 0, 3, "SAFe"),
    (40, 18, 700000,  3, 4, 4, 2, 2, 3, 0, 3, "SAFe"),
    (70, 30, 1200000, 3, 3, 5, 2, 4, 4, 1, 3, "SAFe"),

    # V-Model — medical/automotive, strict verification
    (20, 30, 450000,  5, 1, 4, 2, 3, 4, 1, 1, "V-Model"),
    (15, 24, 350000,  5, 1, 3, 1, 3, 4, 1, 1, "V-Model"),
    (25, 36, 600000,  5, 2, 4, 2, 3, 5, 1, 1, "V-Model"),
    (30, 48, 800000,  5, 1, 5, 2, 3, 5, 1, 1, "V-Model"),
    (18, 24, 400000,  4, 1, 4, 1, 3, 4, 1, 1, "V-Model"),
]

# ---------------------------------------------------------------------------
# SDLC labelling — 10 models
# ---------------------------------------------------------------------------
def _label_sdlc(clarity, involvement, complexity, risk,
                team_size, duration, experience, compliance, geo):

    # Kanban: tiny team, short, simple
    if team_size <= 5 and duration <= 6 and complexity <= 2:
        return "Kanban"

    # V-Model: compliance + frozen reqs + verification-heavy
    if compliance == 1 and clarity >= 4 and risk <= 2 and complexity >= 3:
        return "V-Model"

    # Spiral: high risk + high complexity
    if compliance == 1 and risk >= 2 and complexity >= 4:
        return "Spiral"
    if risk == 3 and complexity >= 4:
        return "Spiral"
    if risk == 3 and complexity == 3 and duration >= 18:
        return "Spiral"

    # SAFe: large enterprise teams, scaled agile, distributed
    if team_size >= 50 and geo >= 2:
        return "SAFe"
    if team_size >= 70:
        return "SAFe"

    # RAD: short, prototype-driven, low risk
    if duration <= 5 and involvement >= 4 and risk == 1 and complexity <= 3:
        return "RAD"
    if duration <= 4 and involvement >= 3 and risk <= 2:
        return "RAD"

    # XP: small expert team, high involvement, high complexity
    if team_size <= 6 and experience >= 4 and involvement >= 4 and complexity >= 4:
        return "XP"
    if team_size <= 5 and experience == 5 and involvement >= 3:
        return "XP"

    # Waterfall: frozen reqs, low risk
    if clarity >= 4 and risk == 1:
        return "Waterfall"
    if clarity == 5 and risk <= 2 and involvement <= 2:
        return "Waterfall"

    # Scrum: high involvement, evolving reqs
    if involvement >= 4 and clarity <= 3:
        return "Scrum"
    if involvement >= 4 and complexity >= 3 and risk <= 2:
        return "Scrum"
    if geo == 3 and involvement >= 3:
        return "Scrum"

    # Iterative: vague reqs, low involvement
    if clarity <= 2 and involvement <= 2:
        return "Iterative"
    if clarity == 1 and risk >= 2:
        return "Iterative"
    if experience <= 2 and clarity <= 2:
        return "Iterative"

    return "Agile"


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------
def inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n  = len(df)

    for col in ["budget_usd", "budget_per_person"]:
        df.loc[:, col] = (df[col] * rng.normal(1.0, 0.03, n)).clip(lower=1).round(2)

    for col, (lo, hi) in {
        "requirements_clarity": (1, 5),
        "client_involvement":   (1, 5),
        "tech_complexity":      (1, 5),
        "team_experience":      (1, 5),
    }.items():
        df.loc[:, col] = df[col].add(rng.integers(-1, 2, n)).clip(lo, hi)

    flip_mask = rng.random(n) < 0.02
    if flip_mask.sum() > 0:
        alts = {
            "Waterfall": ["Agile", "V-Model"],
            "Agile":     ["Scrum", "Iterative"],
            "Scrum":     ["Agile", "XP"],
            "Kanban":    ["Agile", "Scrum"],
            "Spiral":    ["Iterative", "V-Model"],
            "Iterative": ["Agile", "Scrum"],
            "RAD":       ["Agile", "Scrum"],
            "XP":        ["Agile", "Scrum"],
            "SAFe":      ["Agile", "Scrum"],
            "V-Model":   ["Waterfall", "Spiral"],
        }
        df.loc[flip_mask, "sdlc"] = [
            rng.choice(alts.get(s, ["Agile"]))
            for s in df.loc[flip_mask, "sdlc"].values
        ]

    om = rng.random(n) < 0.01
    df.loc[om, "team_size"]       = rng.integers(40, 81, om.sum())
    df.loc[om, "duration_months"] = rng.integers(36, 49, om.sum())
    df.loc[om, "budget_usd"]      = rng.uniform(800_000, 1_500_000, om.sum()).round(2)
    df.loc[:, "budget_per_person"] = (df["budget_usd"] / df["team_size"].clip(lower=1)).round(4)
    df.loc[:, "complexity_risk"]   = df["tech_complexity"] * df["risk_encoded"]
    return df


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(n: int = 10000) -> pd.DataFrame:
    rng  = np.random.default_rng(42)
    rows = []

    for _ in range(n):
        team_size               = int(rng.integers(2, 81))
        duration_months         = int(rng.integers(1, 49))
        budget_usd              = round(float(rng.uniform(10_000, 1_500_000)), 2)
        requirements_clarity    = int(rng.integers(1, 6))
        client_involvement      = int(rng.integers(1, 6))
        tech_complexity         = int(rng.integers(1, 6))
        risk_encoded            = int(rng.integers(1, 4))
        type_encoded            = int(rng.integers(1, 5))
        team_experience         = int(rng.integers(1, 6))
        regulatory_compliance   = int(rng.integers(0, 2))
        geographic_distribution = int(rng.integers(1, 4))

        budget_per_person = round(budget_usd / max(team_size, 1), 4)
        complexity_risk   = tech_complexity * risk_encoded

        sdlc = _label_sdlc(
            requirements_clarity, client_involvement,
            tech_complexity, risk_encoded,
            team_size, duration_months,
            team_experience, regulatory_compliance, geographic_distribution,
        )

        utilisation = 0.55 + (tech_complexity - 1) * 0.05
        effort_base = team_size * duration_months * utilisation
        effort = (effort_base
                  * (1 + (risk_encoded - 1) * 0.15)
                  * (1 + TYPE_COMPLEXITY_BOOST[type_encoded])
                  * EXPERIENCE_EFFORT_FACTOR[team_experience]
                  * GEO_EFFORT_FACTOR[geographic_distribution])
        effort = round(float(effort + rng.normal(0, effort * 0.05)), 2)
        effort = max(effort, 1.0)

        base_rate = 5_000 + (tech_complexity - 1) * 2_500
        cost_base = (effort * base_rate
                     * SDLC_COST_FACTOR[sdlc]
                     * COMPLIANCE_COST_FACTOR[regulatory_compliance])
        cost_base = min(cost_base, budget_usd * 1.2)
        cost = round(float(max(cost_base + rng.normal(0, cost_base * 0.08), 5_000)), 2)

        rows.append([
            team_size, duration_months, budget_usd,
            requirements_clarity, client_involvement, tech_complexity,
            risk_encoded, type_encoded, budget_per_person, complexity_risk,
            team_experience, regulatory_compliance, geographic_distribution,
            sdlc, effort, cost,
        ])

    df_syn = pd.DataFrame(rows, columns=FEATURE_COLS + ["sdlc", "effort_person_months", "cost_usd"])

    real_rows = []
    for (team, dur, budget, clarity, involv, complex_, risk, ptype,
         exp, comply, geo, sdlc) in REAL_WORLD_SEEDS:
        bpp    = round(budget / max(team, 1), 4)
        cr     = complex_ * risk
        effort = round(
            team * dur * (0.55 + (complex_ - 1) * 0.05)
            * (1 + (risk - 1) * 0.15)
            * (1 + TYPE_COMPLEXITY_BOOST[ptype])
            * EXPERIENCE_EFFORT_FACTOR[exp]
            * GEO_EFFORT_FACTOR[geo], 2
        )
        cost = round(max(
            effort * (5_000 + (complex_ - 1) * 2_500)
            * SDLC_COST_FACTOR[sdlc]
            * COMPLIANCE_COST_FACTOR[comply],
            5_000
        ), 2)
        real_rows.append([team, dur, budget, clarity, involv, complex_,
                          risk, ptype, bpp, cr, exp, comply, geo,
                          sdlc, effort, cost])

    df_real     = pd.DataFrame(real_rows, columns=FEATURE_COLS + ["sdlc", "effort_person_months", "cost_usd"])
    df_real_aug = inject_noise(pd.concat([df_real] * 5, ignore_index=True), rng)
    df_syn      = inject_noise(df_syn, rng)

    return pd.concat([df_syn, df_real_aug], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df):
    X = df[FEATURE_COLS].copy()
    for col in ["budget_usd", "budget_per_person"]:
        lo, hi = X[col].quantile([0.01, 0.99])
        X.loc[:, col] = X[col].clip(lo, hi)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le       = LabelEncoder()
    y_clf    = le.fit_transform(df["sdlc"].values)
    y_effort = df["effort_person_months"].values.astype(float)
    y_cost   = df["cost_usd"].values.astype(float)
    return X_scaled, y_clf, y_effort, y_cost, scaler, le


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _reg_metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    print(f"\n── {label} ──")
    print(f"  MAE  : {mae:>12,.2f}  |  RMSE : {rmse:>12,.2f}  |  MAPE : {mape:.1f}%  |  R² : {r2:.4f}")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2), "r2": round(r2, 4)}


def _clf_metrics(y_true, y_pred, classes, label):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    print(f"\n── {label} ──")
    print(f"  Accuracy : {acc:.2%}  |  F1 (wtd) : {f1:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=classes))
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    df = generate_dataset(10000)
    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset  : {len(df):,} rows  |  {len(FEATURE_COLS)} features")
    print("SDLC distribution:")
    for sdlc, cnt in df["sdlc"].value_counts().items():
        print(f"  {sdlc:<12} {cnt:>5}  {'█' * int(cnt/60)}")

    X, y_clf, y_effort, y_cost, scaler, le = preprocess(df)

    (X_train, X_test,
     yc_train, yc_test,
     ye_train, ye_test,
     yr_train, yr_test) = train_test_split(
        X, y_clf, y_effort, y_cost,
        test_size=0.2, random_state=42, stratify=y_clf,
    )
    print(f"\nTrain : {len(X_train):,}  |  Test : {len(X_test):,}")

    cost_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42)
    cost_model.fit(X_train, yr_train)
    cost_metrics = _reg_metrics(yr_test, cost_model.predict(X_test), "Cost Estimator")

    effort_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42)
    effort_model.fit(X_train, ye_train)
    effort_metrics = _reg_metrics(ye_test, effort_model.predict(X_test), "Effort Estimator")

    clf_model = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=1)
    clf_model.fit(X_train, yc_train)
    clf_metrics = _clf_metrics(yc_test, clf_model.predict(X_test), list(le.classes_), "SDLC Classifier")

    cv = cross_val_score(clf_model, X, y_clf, cv=5, scoring="accuracy", n_jobs=1)
    print(f"  CV Accuracy : {cv.mean():.2%} ± {cv.std():.4f}")
    clf_metrics["cv_accuracy_mean"] = round(cv.mean(), 4)
    clf_metrics["cv_accuracy_std"]  = round(cv.std(), 4)

    fi = pd.Series(clf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    for feat, imp in fi.items():
        print(f"  {feat:<28} {imp:.4f}  {'█' * int(imp * 60)}")

    for fname, obj in [
        ("scaler.pkl",        scaler),
        ("cost_reg.pkl",      cost_model),
        ("effort_reg.pkl",    effort_model),
        ("sdlc_clf.pkl",      clf_model),
        ("label_encoder.pkl", le),
    ]:
        with open(os.path.join(MODEL_DIR, fname), "wb") as f:
            pickle.dump(obj, f)

    metrics = {
        "cost_estimator":     cost_metrics,
        "effort_estimator":   effort_metrics,
        "sdlc_classifier":    clf_metrics,
        "feature_importance": fi.round(4).to_dict(),
        "sdlc_distribution":  df["sdlc"].value_counts().to_dict(),
        "dataset_size":       len(df),
        "feature_cols":       FEATURE_COLS,
        "sdlc_labels":        list(le.classes_),
    }
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    print("\n✓ All artifacts saved to", MODEL_DIR)
    return metrics


if __name__ == "__main__":
    train()
