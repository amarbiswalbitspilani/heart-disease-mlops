# Stage 1: train the model from the committed data, so a fresh clone builds without extra steps.
FROM python:3.11.17-slim AS train
WORKDIR /build
# Longer timeout and more retries: package downloads on slow or flaky networks.
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml ./
COPY src/ src/
COPY data/ data/
RUN PYTHONPATH=src python -m heart.train

# Stage 2: slim serving image with only the runtime dependencies and the trained model.
FROM python:3.11.17-slim AS serve
LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI)"
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt && useradd --create-home --uid 10001 appuser
COPY src/heart/ src/heart/
COPY api/ api/
COPY --from=train /build/models/model.joblib models/model.joblib
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
# One worker per pod: Kubernetes scales by replicas. Access logs come from the app as JSON.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
