"""FastAPI service that serves the heart disease model.

Endpoints:
  POST /predict  patient features (JSON) -> prediction, probability and confidence
  GET  /health   liveness/readiness check used by Docker and Kubernetes
  GET  /metrics  Prometheus metrics (request count, latency, predictions, confidence)

Every request is logged as one JSON line. Patient feature values are never logged.
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import PatientFeatures, Prediction
from heart.config import FEATURES
from heart.config import MODEL_PATH as DEFAULT_MODEL_PATH

MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

REQUESTS = Counter(
    "heartapi_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)  # fmt: skip
LATENCY = Histogram(
    "heartapi_request_latency_seconds", "Request latency in seconds", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5),
)  # fmt: skip
PREDICTIONS = Counter(
    "heartapi_predictions_total", "Predictions by class", ["label", "model_version"]
)  # fmt: skip
CONFIDENCE = Histogram(
    "heartapi_prediction_confidence", "Probability of the predicted class",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)  # fmt: skip


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


log = logging.getLogger("heart_api")
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def model_version(path: Path) -> str:
    digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    return os.environ.get("MODEL_VERSION", f"1.0.0+{digest}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # joblib (pickle) runs code on load, so only the model file produced by this project's own
    # training step (inside the image build or the repo) is ever loaded; no uploaded files.
    app.state.model = joblib.load(MODEL_PATH)
    app.state.version = model_version(MODEL_PATH)
    log.info("model loaded", extra={"fields": {"model_version": app.state.version}})
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description="Predicts the presence of heart disease from 13 patient attributes "
    "(UCI Heart Disease, Cleveland).",
    lifespan=lifespan,
)


REQUEST_ID = re.compile(r"[A-Za-z0-9-]{1,64}")
NOT_COUNTED = ("/health", "/metrics")  # probes and scrapes are not API traffic
METHODS = {"GET", "POST", "HEAD"}  # any other method is labelled "other"


@app.middleware("http")
async def observe(request: Request, call_next):
    given = request.headers.get("x-request-id", "")
    request_id = given if REQUEST_ID.fullmatch(given) else uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"fields": {"request_id": request_id}})
        response = JSONResponse(
            {"detail": "Internal server error", "request_id": request_id}, status_code=500
        )
    response.headers["x-request-id"] = request_id
    elapsed = time.perf_counter() - start
    # Label by route template (e.g. /predict), never the raw path, so unknown URLs cannot
    # create new metric series.
    route = request.scope.get("route")
    docs = {app.docs_url, app.redoc_url, app.openapi_url, app.swagger_ui_oauth2_redirect_url}
    if route is not None:
        endpoint = route.path
    else:  # FastAPI's documentation pages are served without a route object
        endpoint = request.url.path if request.url.path in docs else "unmatched"
    method = request.method if request.method in METHODS else "other"
    if endpoint not in NOT_COUNTED:
        REQUESTS.labels(method, endpoint, str(response.status_code)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        fields = {
            "request_id": request_id,
            "method": method,
            "path": endpoint,
            "status": response.status_code,
            "latency_ms": round(elapsed * 1000, 2),
        }
        fields.update(getattr(request.state, "log_fields", {}))
        level = logging.ERROR if response.status_code >= 500 else logging.INFO
        log.log(level, "request", extra={"fields": fields})
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": app.state.version}


@app.post("/predict", response_model=Prediction)
def predict(patient: PatientFeatures, request: Request) -> Prediction:
    row = pd.DataFrame([patient.to_features()])[FEATURES].astype(float)
    p = float(app.state.model.predict_proba(row)[0, 1])
    label = int(p >= 0.5)
    confidence = p if label else 1 - p
    PREDICTIONS.labels("disease" if label else "no disease", app.state.version).inc()
    CONFIDENCE.observe(confidence)
    request.state.log_fields = {"prediction": label, "confidence": round(confidence, 4)}
    return Prediction(
        prediction=label,
        label="disease" if label else "no disease",
        probability_of_disease=round(p, 4),
        confidence=round(confidence, 4),
        model_version=app.state.version,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
