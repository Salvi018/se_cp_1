import os
import pickle
from flask import Blueprint, jsonify

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/models/metrics", methods=["GET"])
def model_metrics():
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "metrics.pkl")
        with open(path, "rb") as f:
            metrics = pickle.load(f)
        return jsonify(metrics), 200
    except FileNotFoundError:
        return jsonify({"error": "Models not trained yet. Run train.py first."}), 404
