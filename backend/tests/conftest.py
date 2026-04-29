import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app, db as _db

@pytest.fixture(scope="session")
def app():
    os.environ["FLASK_ENV"]      = "testing"
    os.environ["DATABASE_URL"]   = "sqlite:///:memory:"
    os.environ["SECRET_KEY"]     = "test-secret"
    app = create_app()
    app.config["TESTING"] = True
    with app.app_context():
        _db.create_all()
        yield app

@pytest.fixture(scope="session")
def client(app):
    return app.test_client()

# Valid full payload used across multiple tests
@pytest.fixture
def valid_payload():
    return {
        "team_size":               10,
        "duration_months":         12,
        "budget_usd":              200000,
        "requirements_clarity":    3,
        "client_involvement":      4,
        "tech_complexity":         3,
        "risk_level":              "medium",
        "project_type":            "web",
        "team_experience":         3,
        "regulatory_compliance":   0,
        "geographic_distribution": 1,
    }
