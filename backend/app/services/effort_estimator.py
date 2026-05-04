import os
import pickle
import numpy as np
import pandas as pd

_cost_model = _effort_model = _scaler = None

# Feature columns matching training
FEATURE_COLS = [
    "team_size", "duration_months", "budget_usd",
    "requirements_clarity", "client_involvement", "tech_complexity",
    "risk_encoded", "type_encoded", "budget_per_person", "complexity_risk",
    "team_experience", "regulatory_compliance", "geographic_distribution",
    # New engineered features
    "team_efficiency", "project_complexity_index", "budget_efficiency",
    "experience_complexity_ratio", "involvement_clarity_product",
]


def _load():
    global _cost_model, _effort_model, _scaler
    if _cost_model is None:
        d = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
        with open(os.path.join(d, "cost_reg.pkl"),   "rb") as f: _cost_model   = pickle.load(f)
        with open(os.path.join(d, "effort_reg.pkl"), "rb") as f: _effort_model = pickle.load(f)
        with open(os.path.join(d, "scaler.pkl"),     "rb") as f: _scaler       = pickle.load(f)


MIN_COST_USD = 5_000   # same floor applied during training

def estimate_cost(feature_vector: np.ndarray, **_) -> dict:
    _load()
    # Create DataFrame with feature names for scaler compatibility
    X_df = pd.DataFrame(feature_vector, columns=FEATURE_COLS)
    X_scaled = _scaler.transform(X_df)

    raw_cost = float(_cost_model.predict(X_scaled)[0])
    raw_effort = float(_effort_model.predict(X_scaled)[0])

    # Hard floors — cost and effort can never be negative
    cost   = max(raw_cost,   MIN_COST_USD)   # minimum $5,000
    effort = max(raw_effort, 0.5)            # minimum 0.5 person-months

    margin = cost * 0.20   # compute margin on floored cost so bounds are always valid

    return {
        "effort_person_months": round(effort, 2),
        "estimated_cost_usd":   round(cost, 2),
        "cost_lower":           round(max(cost - margin, MIN_COST_USD), 2),
        "cost_upper":           round(cost + margin, 2),
    }
