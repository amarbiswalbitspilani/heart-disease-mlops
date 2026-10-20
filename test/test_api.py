import io
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.app as service

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "api" / "sample_input.json").read_text())


@pytest.fixture()
def client(model_file, monkeypatch):
    monkeypatch.setattr(service, "MODEL_PATH", model_file)
    with TestClient(service.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("disease" if body["prediction"] else "no disease")
    assert 0 <= body["probability_of_disease"] <= 1
    assert body["confidence"] == round(max(body["probability_of_disease"],
                                           1 - body["probability_of_disease"]), 4)  # fmt: skip


def test_unknown_values_are_accepted_as_null(client):
    r = client.post("/predict", json={**SAMPLE, "major_vessels": None, "thalassemia": None})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"age": 5},
        {"sex": 2},
        {"chest_pain_type": 0},
        {"cholesterol": -1},
        {"extra_field": 1},
        {"sex": True},
        {"exercise_induced_angina": False},
    ],
)
def test_invalid_input_is_rejected(client, change):
    assert client.post("/predict", json={**SAMPLE, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/predict", json=body).status_code == 422


def test_metrics_exposes_request_and_prediction_counters(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "heartapi_requests_total" in text
    assert "heartapi_predictions_total" in text
    assert "heartapi_request_latency_seconds_bucket" in text


def test_unknown_paths_share_one_metric_label(client):
    client.get("/no/such/path")
    client.get("/another/unknown")
    text = client.get("/metrics").text
    assert 'endpoint="unmatched"' in text
    assert "/no/such/path" not in text and "/another/unknown" not in text


def test_health_probes_are_not_counted_as_traffic(client):
    client.get("/health")
    text = client.get("/metrics").text
    assert 'endpoint="/health"' not in text


def test_server_error_returns_json_with_request_id_and_logs_error(client, monkeypatch):
    class Broken:
        def predict_proba(self, _):
            raise RuntimeError("model failure")

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(service.JsonFormatter())
    service.log.addHandler(handler)
    monkeypatch.setattr(service.app.state, "model", Broken())
    try:
        r = client.post("/predict", json=SAMPLE, headers={"x-request-id": "trace-123"})
    finally:
        service.log.removeHandler(handler)
    assert r.status_code == 500
    assert r.json()["request_id"] == "trace-123" == r.headers["x-request-id"]
    errors = [e for e in map(json.loads, stream.getvalue().splitlines()) if e["level"] == "ERROR"]
    assert any("model failure" in e.get("exception", "") for e in errors)
    assert any(e.get("status") == 500 and e.get("request_id") == "trace-123" for e in errors)


def test_unsafe_request_id_is_replaced(client):
    r = client.get("/health", headers={"x-request-id": "x" * 500})
    assert len(r.headers["x-request-id"]) <= 64


def test_unknown_methods_and_docs_have_bounded_labels(client):
    client.request("FOO", "/predict")
    client.get("/docs")
    text = client.get("/metrics").text
    assert 'method="other"' in text and 'method="FOO"' not in text
    assert 'endpoint="/docs"' in text
