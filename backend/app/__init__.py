from flask import Flask, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config.from_object("config.Config")

    # CORS — allow all origins in dev; lock down via CORS_ORIGINS env var in prod
    CORS(app, resources={r"/api/*": {"origins": "*"}},
         supports_credentials=False)

    db.init_app(app)

    # ── Blueprints ────────────────────────────────────────────────────────────
    from app.routes.predict import predict_bp
    from app.routes.history import history_bp
    from app.routes.metrics import metrics_bp
    from app.routes.charts  import charts_bp

    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(history_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api")
    app.register_blueprint(charts_bp,  url_prefix="/api")

    # ── Global error handlers ─────────────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    with app.app_context():
        db.create_all()

    return app
