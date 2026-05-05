"""Smoke tests for Flask API. Pipeline runs in stub mode without trained weights."""
import base64
import io
import os

os.environ.setdefault("FLASK_DEBUG", "1")  # bypass SECRET_KEY guard during tests
os.environ.setdefault("RATELIMIT_ENABLED", "False")

import numpy as np
import pytest
from PIL import Image

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["RATELIMIT_ENABLED"] = False
    with app.test_client() as c:
        yield c


def _fake_frame_b64(w=320, h=240):
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert "active_sessions" in body


def test_predict_missing_frame(client):
    res = client.post("/api/predict", json={})
    assert res.status_code == 400


def test_predict_invalid_frame(client):
    res = client.post("/api/predict", json={"frame": "not-base64"})
    assert res.status_code == 400


def test_predict_blank_frame(client):
    res = client.post("/api/predict", json={
        "frame": _fake_frame_b64(),
        "session_id": "test-session",
    })
    assert res.status_code == 200
    body = res.get_json()
    assert "buffer_size" in body
    assert "sentence" in body
    assert "latency_ms" in body


def test_reset(client):
    res = client.post("/api/reset", json={"session_id": "test-session"})
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"
