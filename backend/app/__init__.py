import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")


def create_app():
    app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="")
    app.config.from_object("config.Config")

    CORS(app, resources={r"/api/*": {"origins": "*"}})
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

    # ── Serve React SPA ───────────────────────────────────────────────────────
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        # For /api/ paths, check the URL map and return the correct status code
        # so Flask's method-not-allowed (405) semantics work properly.
        if path.startswith("api/"):
            from flask import request as req
            adapter = app.url_map.bind("")
            try:
                adapter.match("/" + path, method=req.method)
            except Exception as exc:
                exc_name = type(exc).__name__
                if exc_name == "MethodNotAllowed":
                    return jsonify({"error": "Method not allowed"}), 405
                return jsonify({"error": "Not found"}), 404
        full = os.path.join(FRONTEND_DIST, path)
        if path and os.path.exists(full):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    # ── Global error handlers ─────────────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        from flask import request as req
        # API paths get a JSON 404; SPA paths get index.html for React Router
        if req.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(FRONTEND_DIST, "index.html")

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    with app.app_context():
        db.create_all()

    return app
