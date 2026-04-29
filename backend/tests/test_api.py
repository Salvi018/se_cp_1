"""
Integration Tests — Frontend ↔ Backend API
==========================================
Tests every endpoint the React frontend calls.
"""

import pytest


# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------
class TestPredict:

    def test_valid_request_returns_200(self, client, valid_payload):
        r = client.post("/api/predict", json=valid_payload)
        assert r.status_code == 200

    def test_response_has_success_true(self, client, valid_payload):
        r = client.post("/api/predict", json=valid_payload)
        assert r.get_json()["success"] is True

    def test_response_contains_all_fields(self, client, valid_payload):
        data = client.post("/api/predict", json=valid_payload).get_json()["data"]
        assert "recommended_sdlc"     in data
        assert "confidence"           in data
        assert "alternatives"         in data
        assert "effort_person_months" in data
        assert "estimated_cost_usd"   in data
        assert "cost_range"           in data
        assert "lower"                in data["cost_range"]
        assert "upper"                in data["cost_range"]

    def test_recommended_sdlc_is_valid(self, client, valid_payload):
        VALID = {"Waterfall","Agile","Scrum","Kanban","Spiral",
                 "Iterative","RAD","XP","SAFe","V-Model"}
        data = client.post("/api/predict", json=valid_payload).get_json()["data"]
        assert data["recommended_sdlc"] in VALID

    def test_confidence_between_0_and_1(self, client, valid_payload):
        data = client.post("/api/predict", json=valid_payload).get_json()["data"]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_cost_range_lower_less_than_upper(self, client, valid_payload):
        data = client.post("/api/predict", json=valid_payload).get_json()["data"]
        assert data["cost_range"]["lower"] < data["cost_range"]["upper"]

    def test_effort_is_positive(self, client, valid_payload):
        data = client.post("/api/predict", json=valid_payload).get_json()["data"]
        assert data["effort_person_months"] > 0

    def test_alternatives_is_list(self, client, valid_payload):
        data = client.post("/api/predict", json=valid_payload).get_json()["data"]
        assert isinstance(data["alternatives"], list)

    def test_missing_required_field_returns_422(self, client):
        r = client.post("/api/predict", json={"team_size": 10})
        assert r.status_code == 422
        body = r.get_json()
        assert body["success"] is False
        assert "details" in body

    def test_invalid_team_size_returns_422(self, client, valid_payload):
        payload = {**valid_payload, "team_size": 999}
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_risk_level_returns_422(self, client, valid_payload):
        payload = {**valid_payload, "risk_level": "extreme"}
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_json_returns_400(self, client):
        r = client.post("/api/predict", data="not-json",
                        content_type="text/plain")
        assert r.status_code == 400

    def test_wrong_method_returns_405(self, client):
        # With SPA catch-all, GET /api/predict returns 405
        r = client.get("/api/predict")
        assert r.status_code in (405, 404)

    def test_backward_compatible_without_new_fields(self, client):
        """Old 8-field requests must still work."""
        payload = {
            "team_size": 10, "duration_months": 12, "budget_usd": 200000,
            "requirements_clarity": 3, "client_involvement": 4,
            "tech_complexity": 3, "risk_level": "medium", "project_type": "web",
        }
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200

    @pytest.mark.parametrize("profile,expected_sdlc", [
        # (payload_overrides, expected)
        ({"team_size": 3,  "duration_months": 4,  "requirements_clarity": 3,
          "client_involvement": 4, "tech_complexity": 1, "risk_level": "low",
          "team_experience": 3, "regulatory_compliance": 0, "geographic_distribution": 1},
         "Kanban"),
        ({"team_size": 20, "duration_months": 30, "requirements_clarity": 3,
          "client_involvement": 2, "tech_complexity": 5, "risk_level": "high",
          "team_experience": 5, "regulatory_compliance": 1, "geographic_distribution": 1,
          "project_type": "embedded"},
         "Spiral"),
        ({"team_size": 60, "duration_months": 24, "requirements_clarity": 4,
          "client_involvement": 3, "tech_complexity": 4, "risk_level": "medium",
          "team_experience": 4, "regulatory_compliance": 0, "geographic_distribution": 3},
         "SAFe"),
    ])
    def test_sdlc_profiles(self, client, valid_payload, profile, expected_sdlc):
        payload = {**valid_payload, **profile}
        data = client.post("/api/predict", json=payload).get_json()["data"]
        assert data["recommended_sdlc"] == expected_sdlc


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------
class TestHistory:

    def test_returns_200(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200

    def test_response_structure(self, client):
        body = client.get("/api/history").get_json()
        assert "predictions" in body
        assert "total"       in body
        assert "page"        in body
        assert "pages"       in body

    def test_predictions_is_list(self, client):
        body = client.get("/api/history").get_json()
        assert isinstance(body["predictions"], list)

    def test_pagination_params(self, client):
        r = client.get("/api/history?page=1&per_page=5")
        assert r.status_code == 200

    def test_history_grows_after_predict(self, client, valid_payload):
        before = client.get("/api/history").get_json()["total"]
        client.post("/api/predict", json=valid_payload)
        after  = client.get("/api/history").get_json()["total"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# GET /api/models/metrics
# ---------------------------------------------------------------------------
class TestMetrics:

    def test_returns_200(self, client):
        r = client.get("/api/models/metrics")
        assert r.status_code == 200

    def test_contains_classifier_metrics(self, client):
        body = client.get("/api/models/metrics").get_json()
        assert "sdlc_classifier"  in body
        assert "accuracy"         in body["sdlc_classifier"]
        assert "f1_weighted"      in body["sdlc_classifier"]

    def test_contains_cost_metrics(self, client):
        body = client.get("/api/models/metrics").get_json()
        assert "cost_estimator" in body
        assert "mae"            in body["cost_estimator"]
        assert "r2"             in body["cost_estimator"]

    def test_contains_effort_metrics(self, client):
        body = client.get("/api/models/metrics").get_json()
        assert "effort_estimator" in body
        assert "mae"              in body["effort_estimator"]

    def test_accuracy_is_reasonable(self, client):
        body = client.get("/api/models/metrics").get_json()
        acc  = body["sdlc_classifier"]["accuracy"]
        assert 0.5 <= acc <= 1.0   # at least 50% accuracy


# ---------------------------------------------------------------------------
# GET /api/charts/feature-importance
# ---------------------------------------------------------------------------
class TestFeatureImportance:

    def test_returns_200(self, client):
        r = client.get("/api/charts/feature-importance")
        assert r.status_code == 200

    def test_returns_13_features(self, client):
        data = client.get("/api/charts/feature-importance").get_json()["data"]
        assert len(data) == 13

    def test_importances_sum_to_1(self, client):
        data  = client.get("/api/charts/feature-importance").get_json()["data"]
        total = sum(d["importance"] for d in data)
        assert abs(total - 1.0) < 0.01

    def test_each_item_has_required_keys(self, client):
        data = client.get("/api/charts/feature-importance").get_json()["data"]
        for item in data:
            assert "feature"    in item
            assert "importance" in item
            assert "pct"        in item


# ---------------------------------------------------------------------------
# GET /api/charts/predicted-vs-actual
# ---------------------------------------------------------------------------
class TestPredictedVsActual:

    def test_returns_200(self, client):
        r = client.get("/api/charts/predicted-vs-actual")
        assert r.status_code == 200

    def test_has_cost_and_effort(self, client):
        data = client.get("/api/charts/predicted-vs-actual").get_json()["data"]
        assert "cost"   in data
        assert "effort" in data

    def test_cost_points_not_empty(self, client):
        data = client.get("/api/charts/predicted-vs-actual").get_json()["data"]
        assert len(data["cost"]["points"]) > 0

    def test_each_point_has_actual_and_predicted(self, client):
        data   = client.get("/api/charts/predicted-vs-actual").get_json()["data"]
        points = data["cost"]["points"]
        for pt in points[:5]:
            assert "actual"    in pt
            assert "predicted" in pt
