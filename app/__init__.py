import os

from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import Config, _DEV_SECRET_KEY
from app.pipeline import Pipeline
from app.utils.audio_cleanup import start_cleanup_thread
from app.utils.logger import log


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT_DEFAULT],
    headers_enabled=True,
    storage_uri="memory://",
)


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    if (
        not app.config.get("DEBUG")
        and not app.config.get("TESTING")
        and app.config.get("SECRET_KEY") == _DEV_SECRET_KEY
    ):
        raise RuntimeError(
            "SECRET_KEY env var is required when FLASK_DEBUG=0. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    cors_origins = os.getenv("CORS_ORIGINS", "*")
    origins = [o.strip() for o in cors_origins.split(",")] if cors_origins != "*" else "*"
    CORS(app, resources={r"/api/*": {"origins": origins}})

    limiter.init_app(app)

    log.info("initializing pipeline...")
    app.pipeline = Pipeline(app.config)
    log.info("pipeline ready")

    start_cleanup_thread(
        directory=app.config["TTS_OUTPUT_DIR"],
        max_age_seconds=600,
        interval=300,
    )

    from app.routes.main import bp as main_bp
    from app.routes.api import bp as api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(413)
    def payload_too_large(_):
        return jsonify({"error": "payload too large"}), 413

    @app.errorhandler(429)
    def rate_limited(exc):
        return jsonify({"error": "rate limit exceeded", "detail": str(exc.description)}), 429

    @app.errorhandler(500)
    def server_error(exc):
        log.exception(f"500 error: {exc}")
        body = {"error": "internal server error"}
        if app.config.get("DEBUG"):
            body["detail"] = str(exc)
        return jsonify(body), 500

    return app
