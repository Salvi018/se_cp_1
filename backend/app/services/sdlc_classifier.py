import os
import pickle
import numpy as np

_clf = _scaler = _le = None


def _load():
    global _clf, _scaler, _le
    if _clf is None:
        d = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
        with open(os.path.join(d, "sdlc_clf.pkl"),      "rb") as f: _clf    = pickle.load(f)
        with open(os.path.join(d, "scaler.pkl"),         "rb") as f: _scaler = pickle.load(f)
        with open(os.path.join(d, "label_encoder.pkl"),  "rb") as f: _le     = pickle.load(f)


def predict_sdlc(feature_vector: np.ndarray) -> dict:
    _load()
    X       = _scaler.transform(feature_vector)
    proba   = _clf.predict_proba(X)[0]
    indices = np.argsort(proba)[::-1]
    classes = _le.classes_

    return {
        "recommended_sdlc": classes[indices[0]],
        "confidence":       round(float(proba[indices[0]]), 4),
        "alternatives":     [classes[i] for i in indices[1:4]],
    }
