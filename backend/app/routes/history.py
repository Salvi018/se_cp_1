from flask import Blueprint, request, jsonify
from app.models.db import Prediction

history_bp = Blueprint("history", __name__)

@history_bp.route("/history", methods=["GET"])
def history():
    page     = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    paginated = Prediction.query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return jsonify({
        "total":       paginated.total,
        "page":        paginated.page,
        "pages":       paginated.pages,
        "predictions": [p.to_dict() for p in paginated.items],
    }), 200
