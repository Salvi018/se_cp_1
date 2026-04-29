"""
Model Persistence Tests
=======================
Verifies all .pkl files load correctly and produce valid outputs.
"""

import os
import pickle
import pytest
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "ml", "models")

REQUIRED_FILES = [
    "scaler.pkl",
    "sdlc_clf.pkl",
    "cost_reg.pkl",
    "effort_reg.pkl",
    "label_encoder.pkl",
    "metrics.pkl",
]

VALID_SDLC = {"Waterfall","Agile","Scrum","Kanban","Spiral",
              "Iterative","RAD","XP","SAFe","V-Model"}

SAMPLE_INPUT = np.array([[
    10, 12, 200000, 3, 4, 3, 2, 1, 20000, 6, 3, 0, 1
]])


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------
class TestModelFilesExist:

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_file_exists(self, fname):
        path = os.path.join(MODEL_DIR, fname)
        assert os.path.exists(path), f"{fname} not found — run train.py first"

    @pytest.mark.parametrize("fname", REQUIRED_FILES)
    def test_file_not_empty(self, fname):
        path = os.path.join(MODEL_DIR, fname)
        assert os.path.getsize(path) > 0, f"{fname} is empty"


# ---------------------------------------------------------------------------
# Load & type checks
# ---------------------------------------------------------------------------
class TestModelLoading:

    def _load(self, fname):
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            return pickle.load(f)

    def test_scaler_loads(self):
        from sklearn.preprocessing import StandardScaler
        scaler = self._load("scaler.pkl")
        assert isinstance(scaler, StandardScaler)

    def test_classifier_loads(self):
        from sklearn.ensemble import RandomForestClassifier
        clf = self._load("sdlc_clf.pkl")
        assert isinstance(clf, RandomForestClassifier)

    def test_cost_regressor_loads(self):
        from sklearn.ensemble import GradientBoostingRegressor
        reg = self._load("cost_reg.pkl")
        assert isinstance(reg, GradientBoostingRegressor)

    def test_effort_regressor_loads(self):
        from sklearn.ensemble import GradientBoostingRegressor
        reg = self._load("effort_reg.pkl")
        assert isinstance(reg, GradientBoostingRegressor)

    def test_label_encoder_loads(self):
        from sklearn.preprocessing import LabelEncoder
        le = self._load("label_encoder.pkl")
        assert isinstance(le, LabelEncoder)

    def test_label_encoder_has_10_classes(self):
        le = self._load("label_encoder.pkl")
        assert len(le.classes_) == 10
        for cls in VALID_SDLC:
            assert cls in le.classes_

    def test_metrics_loads_as_dict(self):
        metrics = self._load("metrics.pkl")
        assert isinstance(metrics, dict)

    def test_metrics_has_required_keys(self):
        metrics = self._load("metrics.pkl")
        assert "sdlc_classifier"  in metrics
        assert "cost_estimator"   in metrics
        assert "effort_estimator" in metrics
        assert "feature_cols"     in metrics
        assert "sdlc_labels"      in metrics


# ---------------------------------------------------------------------------
# Inference checks
# ---------------------------------------------------------------------------
class TestModelInference:

    def _load(self, fname):
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            return pickle.load(f)

    def test_scaler_transforms_correctly(self):
        scaler = self._load("scaler.pkl")
        result = scaler.transform(SAMPLE_INPUT)
        assert result.shape == (1, 13)

    def test_classifier_predicts_valid_sdlc(self):
        scaler = self._load("scaler.pkl")
        clf    = self._load("sdlc_clf.pkl")
        le     = self._load("label_encoder.pkl")
        X      = scaler.transform(SAMPLE_INPUT)
        pred   = clf.predict(X)
        label  = le.inverse_transform(pred)[0]
        assert label in VALID_SDLC

    def test_classifier_returns_probabilities(self):
        scaler = self._load("scaler.pkl")
        clf    = self._load("sdlc_clf.pkl")
        X      = scaler.transform(SAMPLE_INPUT)
        proba  = clf.predict_proba(X)[0]
        assert len(proba) == 10
        assert abs(sum(proba) - 1.0) < 0.001

    def test_cost_regressor_returns_positive(self):
        scaler = self._load("scaler.pkl")
        reg    = self._load("cost_reg.pkl")
        X      = scaler.transform(SAMPLE_INPUT)
        cost   = reg.predict(X)[0]
        assert cost > 0

    def test_effort_regressor_returns_positive(self):
        scaler = self._load("scaler.pkl")
        reg    = self._load("effort_reg.pkl")
        X      = scaler.transform(SAMPLE_INPUT)
        effort = reg.predict(X)[0]
        assert effort > 0

    def test_scaler_has_13_features(self):
        scaler = self._load("scaler.pkl")
        assert scaler.n_features_in_ == 13

    def test_classifier_has_13_features(self):
        clf = self._load("sdlc_clf.pkl")
        assert clf.n_features_in_ == 13
