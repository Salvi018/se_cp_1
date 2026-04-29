import os
import pickle
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify
from sklearn.model_selection import train_test_split

charts_bp = Blueprint("charts", __name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "data", "sdlc_dataset.csv")

FEATURE_COLS = [
    "team_size", "duration_months", "budget_usd",
    "requirements_clarity", "client_involvement", "tech_complexity",
    "risk_encoded", "type_encoded", "budget_per_person", "complexity_risk",
    "team_experience", "regulatory_compliance", "geographic_distribution",
]

FEATURE_LABELS = {
    "requirements_clarity":    "Req. Clarity",
    "client_involvement":      "Client Involvement",
    "complexity_risk":         "Complexity × Risk",
    "risk_encoded":            "Risk Level",
    "team_size":               "Team Size",
    "duration_months":         "Duration",
    "tech_complexity":         "Tech Complexity",
    "budget_per_person":       "Budget / Person",
    "budget_usd":              "Budget",
    "type_encoded":            "Project Type",
    "team_experience":         "Team Experience",
    "regulatory_compliance":   "Regulatory Compliance",
    "geographic_distribution": "Geographic Distribution",
}


def _load_artifacts():
    def _pkl(name):
        with open(os.path.join(MODEL_DIR, name), "rb") as f:
            return pickle.load(f)
    return (
        _pkl("scaler.pkl"),
        _pkl("cost_reg.pkl"),
        _pkl("effort_reg.pkl"),
        _pkl("sdlc_clf.pkl"),
        _pkl("label_encoder.pkl"),
    )


def _get_test_split():
    df = pd.read_csv(DATA_PATH)
    for col in ["budget_usd", "budget_per_person"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df.loc[:, col] = df[col].clip(lo, hi)

    scaler, cost_model, effort_model, clf_model, le = _load_artifacts()

    X      = scaler.transform(df[FEATURE_COLS].values)
    y_clf  = le.transform(df["sdlc"].values)
    y_cost = df["cost_usd"].values.astype(float)
    y_eff  = df["effort_person_months"].values.astype(float)

    _, X_test, _, yc_test, _, yr_test, _, ye_test = train_test_split(
        X, y_clf, y_cost, y_eff,
        test_size=0.2, random_state=42, stratify=y_clf,
    )
    return X_test, yc_test, yr_test, ye_test, cost_model, effort_model, clf_model, le


# ── GET /api/charts/feature-importance ───────────────────────────────────────
@charts_bp.route("/charts/feature-importance", methods=["GET"])
def feature_importance():
    try:
        _, _, _, clf_model, _ = _load_artifacts()
        importances = clf_model.feature_importances_

        data = sorted(
            [
                {
                    "feature": FEATURE_LABELS.get(col, col),
                    "key":     col,
                    "importance": round(float(imp), 4),
                    "pct": round(float(imp) * 100, 2),
                }
                for col, imp in zip(FEATURE_COLS, importances)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
        return jsonify({"success": True, "data": data}), 200

    except FileNotFoundError:
        return jsonify({"error": "Models not trained yet. Run train.py first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── GET /api/charts/predicted-vs-actual ──────────────────────────────────────
@charts_bp.route("/charts/predicted-vs-actual", methods=["GET"])
def predicted_vs_actual():
    try:
        X_test, _, yr_test, ye_test, cost_model, effort_model, _, _ = _get_test_split()

        cost_pred   = cost_model.predict(X_test)
        effort_pred = effort_model.predict(X_test)

        # Sample 120 points evenly for a clean scatter
        idx = np.linspace(0, len(yr_test) - 1, min(120, len(yr_test)), dtype=int)

        cost_points = [
            {"actual": round(float(yr_test[i]), 2), "predicted": round(float(cost_pred[i]), 2)}
            for i in idx
        ]
        effort_points = [
            {"actual": round(float(ye_test[i]), 2), "predicted": round(float(effort_pred[i]), 2)}
            for i in idx
        ]

        # Perfect-prediction line range
        cost_max   = round(float(max(yr_test.max(),   cost_pred.max())),   2)
        effort_max = round(float(max(ye_test.max(), effort_pred.max())), 2)

        return jsonify({
            "success": True,
            "data": {
                "cost":   {"points": cost_points,   "max": cost_max},
                "effort": {"points": effort_points, "max": effort_max},
            }
        }), 200

    except FileNotFoundError:
        return jsonify({"error": "Models not trained yet. Run train.py first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
