import base64
import binascii
import io
import re

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request
from PIL import Image, UnidentifiedImageError

from app import limiter
from app.utils.logger import log

bp = Blueprint("api", __name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _decode_frame(data_url: str, max_pixels: int, max_dim: int):
    """Return (frame_bgr, error). Inspect the image header before fully decoding
    so a small base64 string cannot inflate into a multi-GB pixel buffer."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data_url, validate=False)
    except (binascii.Error, ValueError):
        return None, "invalid base64"

    if not img_bytes:
        return None, "empty frame"

    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                return None, "invalid dimensions"
            if w > max_dim or h > max_dim:
                return None, f"frame dimension exceeds {max_dim}px"
            if w * h > max_pixels:
                return None, f"frame area exceeds {max_pixels}px"
    except (UnidentifiedImageError, OSError):
        return None, "unsupported image format"

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "decode failed"
    return frame, None


def _valid_session_id(sid) -> bool:
    return isinstance(sid, str) and bool(_SESSION_ID_RE.match(sid))


@bp.route("/predict", methods=["POST"])
@limiter.limit(lambda: current_app.config["RATE_LIMIT_PREDICT"])
def predict():
    payload = request.get_json(silent=True) or {}
    frame_data = payload.get("frame")
    session_id = payload.get("session_id", "default")
    mode = payload.get("mode", "word")

    if not _valid_session_id(session_id):
        return jsonify({"error": "Invalid 'session_id'"}), 400

    if mode not in ("word", "letter"):
        return jsonify({"error": "Invalid 'mode': must be 'word' or 'letter'"}), 400

    if not frame_data or not isinstance(frame_data, str):
        return jsonify({"error": "Missing 'frame' field"}), 400

    frame, err = _decode_frame(
        frame_data,
        max_pixels=current_app.config["MAX_FRAME_PIXELS"],
        max_dim=current_app.config["MAX_FRAME_DIMENSION"],
    )
    if frame is None:
        return jsonify({"error": err or "invalid frame data"}), 400

    try:
        result = current_app.pipeline.process_frame(frame, session_id=session_id, mode=mode)
    except Exception as exc:
        log.exception(f"predict failed: {exc}")
        body = {"error": "inference failed"}
        if current_app.config.get("DEBUG"):
            body["detail"] = str(exc)
        return jsonify(body), 500

    return jsonify(result)


@bp.route("/reset", methods=["POST"])
@limiter.limit(lambda: current_app.config["RATE_LIMIT_RESET"])
def reset():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id", "default")
    if not _valid_session_id(session_id):
        return jsonify({"error": "Invalid 'session_id'"}), 400
    current_app.pipeline.reset(session_id)
    return jsonify({"status": "ok"})


@bp.route("/health", methods=["GET"])
@limiter.exempt
def health():
    return jsonify({
        "status": "ok",
        "active_sessions": current_app.pipeline.session_count(),
    })
