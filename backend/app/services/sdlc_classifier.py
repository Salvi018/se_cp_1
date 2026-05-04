import os
import pickle
import numpy as np
import pandas as pd

_clf = _scaler = _le = None

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
    global _clf, _scaler, _le
    if _clf is None:
        d = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
        with open(os.path.join(d, "sdlc_clf.pkl"),      "rb") as f: _clf    = pickle.load(f)
        with open(os.path.join(d, "scaler.pkl"),         "rb") as f: _scaler = pickle.load(f)
        with open(os.path.join(d, "label_encoder.pkl"),  "rb") as f: _le     = pickle.load(f)


def predict_sdlc(feature_vector: np.ndarray) -> dict:
    _load()
    # Create DataFrame with feature names for scaler compatibility
    X_df = pd.DataFrame(feature_vector, columns=FEATURE_COLS)
    X_scaled = _scaler.transform(X_df)
    proba   = _clf.predict_proba(X_scaled)[0]
    indices = np.argsort(proba)[::-1]
    classes = _le.classes_

    return {
        "recommended_sdlc": classes[indices[0]],
        "confidence":       round(float(proba[indices[0]]), 4),
        "alternatives":     [classes[i] for i in indices[1:4]],
    }
