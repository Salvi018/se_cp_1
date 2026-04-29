import os
import pickle
import numpy as np

_cost_model = _effort_model = _scaler = None


def _load():
    global _cost_model, _effort_model, _scaler
    if _cost_model is None:
        d = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
        with open(os.path.join(d, "cost_reg.pkl"),   "rb") as f: _cost_model   = pickle.load(f)
        with open(os.path.join(d, "effort_reg.pkl"), "rb") as f: _effort_model = pickle.load(f)
        with open(os.path.join(d, "scaler.pkl"),     "rb") as f: _scaler       = pickle.load(f)


def estimate_cost(feature_vector: np.ndarray, **_) -> dict:
    _load()
    X      = _scaler.transform(feature_vector)
    cost   = float(_cost_model.predict(X)[0])
    effort = max(float(_effort_model.predict(X)[0]), 0.5)  # floor at 0.5 person-months
    margin = cost * 0.20

    return {
        "effort_person_months": round(effort, 2),
        "estimated_cost_usd":   round(cost, 2),
        "cost_lower":           round(max(cost - margin, 0), 2),
        "cost_upper":           round(cost + margin, 2),
    }
