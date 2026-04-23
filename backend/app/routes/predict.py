from flask import Blueprint, request, jsonify
from app import db
from app.models.db import Prediction
from app.utils.preprocessor import validate_and_parse, build_feature_vector
from app.services.sdlc_classifier import predict_sdlc
from app.services.effort_estimator import estimate_cost

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    # ── 1. Parse JSON body ────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error":   "Request body must be valid JSON with Content-Type: application/json"
        }), 400

    # ── 2. Validate & coerce inputs ───────────────────────────────────────────
    parsed, errors = validate_and_parse(data)
    if errors:
        return jsonify({
            "success": False,
            "error":   "Validation failed",
            "details": errors
        }), 422

    # ── 3. Build feature vector ───────────────────────────────────────────────
    try:
        fv = build_feature_vector(parsed)
    except Exception as e:
        return jsonify({"success": False, "error": "Feature engineering failed", "details": str(e)}), 500

    # ── 4. ML inference ───────────────────────────────────────────────────────
    try:
        sdlc_result = predict_sdlc(fv)
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Models not found. Run train.py first."}), 503
    except Exception as e:
        return jsonify({"success": False, "error": "SDLC prediction failed", "details": str(e)}), 500

    try:
        cost_result = estimate_cost(fv)
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Models not found. Run train.py first."}), 503
    except Exception as e:
        return jsonify({"success": False, "error": "Cost estimation failed", "details": str(e)}), 500

    # ── 5. Persist to DB ──────────────────────────────────────────────────────
    try:
        record = Prediction(
            team_size            = parsed["team_size"],
            duration_months      = parsed["duration_months"],
            budget_usd           = parsed["budget_usd"],
            requirements_clarity = parsed["requirements_clarity"],
            client_involvement   = parsed["client_involvement"],
            tech_complexity      = parsed["tech_complexity"],
            risk_level           = parsed["risk_level"],
            project_type         = parsed["project_type"],
            recommended_sdlc     = sdlc_result["recommended_sdlc"],
            confidence           = sdlc_result["confidence"],
            alternatives         = ",".join(sdlc_result["alternatives"]),
            effort_person_months = cost_result["effort_person_months"],
            estimated_cost_usd   = cost_result["estimated_cost_usd"],
            cost_lower           = cost_result["cost_lower"],
            cost_upper           = cost_result["cost_upper"],
        )
        db.session.add(record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": "Database error", "details": str(e)}), 500

    # ── 6. Response ───────────────────────────────────────────────────────────
    return jsonify({
        "success": True,
        "data": {
            "id":                   record.id,
            "recommended_sdlc":     sdlc_result["recommended_sdlc"],
            "confidence":           sdlc_result["confidence"],
            "alternatives":         sdlc_result["alternatives"],
            "effort_person_months": cost_result["effort_person_months"],
            "estimated_cost_usd":   cost_result["estimated_cost_usd"],
            "cost_range": {
                "lower": cost_result["cost_lower"],
                "upper": cost_result["cost_upper"],
            },
            "input_summary": {
                "team_size":       parsed["team_size"],
                "duration_months": parsed["duration_months"],
                "budget_usd":      parsed["budget_usd"],
                "risk_level":      parsed["risk_level"],
                "project_type":    parsed["project_type"],
            }
        }
    }), 200
