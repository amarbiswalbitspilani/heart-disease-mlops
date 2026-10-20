# Heart Disease Risk Prediction — MLOps pipeline

[![ci](https://github.com/amarbiswalbitspilani/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/amarbiswalbitspilani/heart-disease-mlops/actions/workflows/ci.yml)

A classifier that predicts the presence of heart disease from 13 patient attributes, built and
operated as an end-to-end MLOps pipeline: data preparation and EDA, model tuning with
cross-validation, MLflow experiment tracking and model registry, unit tests and GitHub Actions
CI, a FastAPI service in Docker, deployment on Kubernetes behind a LoadBalancer, and monitoring
with Prometheus and Grafana.

**Dataset:** Heart Disease, UCI Machine Learning Repository (dataset id 45), processed Cleveland
file: 303 patients, 13 features and `num` (disease grade 0–4), turned into a binary target
(0 = absent, 1 = present).

## Repository layout

| Path | Contents |
|---|---|
| `scripts/download_data.py` | downloads the UCI file and writes `data/processed/heart_clean.csv` |
| `src/heart/` | data cleaning, preprocessing pipeline, candidate models, training with MLflow |
| `notebooks/01_eda.ipynb` | exploratory data analysis (executed) |
| `notebooks/02_training.ipynb` | runs the training pipeline and inspects tuning, metrics, plots and the registry (executed) |
| `scripts/predict.py` | scores patients from a JSON file with the saved model |
| `api/` | FastAPI service, request/response schemas, `sample_input.json` |
| `test/` | pytest unit tests for data processing, features, models and the API |
| `Dockerfile` | multi-stage image: trains the model, then serves it |
| `.github/workflows/ci.yml` | lint → tests → training → Docker build and smoke test |
| `k8s/` | Namespace, Deployment (2 replicas, probes) and LoadBalancer Service |
| `monitoring/` | Prometheus and Grafana (provisioned dashboard), applied with `kubectl apply -k` |
| `scripts/draw_architecture.py` | draws the architecture diagram, `reports/figures/architecture.png` |
| `reports/` | figures, `metrics.json`, `mlflow_runs.csv` |
| `screenshots/` | screenshots of the runs, CI, deployment and monitoring used in the report |
| `report/` | the written report (PDF) |

## 1. Set up (Python 3.11)

```bash
git clone https://github.com/amarbiswalbitspilani/heart-disease-mlops.git
cd heart-disease-mlops
# Python 3.11 through uv (no sudo); uv downloads Python 3.11 when it is not installed.
# With python3.11 already on the machine, `python3.11 -m venv .venv` replaces these three lines.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 --seed .venv           # --seed installs pip into the venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data and EDA

```bash
python scripts/download_data.py             # UCI download (falls back to the committed copy)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

The cleaned CSV keeps the few missing values (`ca`, `thal`) on purpose: the model pipeline imputes
them with statistics from the training split only.

## 3. Train, track and register

```bash
PYTHONPATH=src python -m heart.train
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # http://localhost:5000
```

`mlflow ui` runs in the foreground; stop it with Ctrl-C before the next step.

Training runs a grid search with stratified 5-fold cross-validation for logistic regression and
a random forest, tuning and selecting on F2 (recall-weighted). It logs one MLflow run per model
(parameters, train/CV/test metrics, plots, tables, model) with one child run per grid combination,
saves the selected model to `models/model.joblib` and registers it as `heart-disease-classifier`
with the alias `champion`.

The same training, with tables and plots inline:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_training.ipynb
```

## 4. Predict and test

```bash
python scripts/predict.py api/sample_input.json
python -m pytest
ruff check . && ruff format --check .
```

## 5. Run the API

Locally (the server runs in the foreground; send the request from a second terminal, then stop the
server with Ctrl-C before the Docker step, which uses the same port):

```bash
PYTHONPATH=src uvicorn api.app:app --port 8000   # docs at http://localhost:8000/docs
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @api/sample_input.json
```

Prerequisites for sections 5–7 on Ubuntu, if not installed yet (Docker needs sudo; Minikube and
kubectl go to `~/.local/bin`, already on the PATH from section 1):

```bash
mkdir -p ~/.local/bin
sudo apt-get install -y docker.io && sudo usermod -aG docker "$USER"
newgrp docker                                 # a shell with the new docker group; continue in it
curl -Lo /tmp/minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
install /tmp/minikube ~/.local/bin/minikube
curl -Lo /tmp/kubectl "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
install -m 0755 /tmp/kubectl ~/.local/bin/kubectl
docker version && minikube version && kubectl version --client
```

With Docker:

```bash
docker build -t heart-api:1.0.0 .
docker run -d --name heart-api -p 8000:8000 heart-api:1.0.0
curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @api/sample_input.json
```

## 6. Deploy on Kubernetes (Minikube)

```bash
minikube start --driver=docker --cpus=4 --memory=6g
minikube addons enable metallb

# MetalLB address pool: .200-.210 inside Minikube's own network, read from `minikube ip`
NET=$(minikube ip | cut -d. -f1-3)           # e.g. 192.168.49
kubectl -n metallb-system patch cm config --type merge -p \
  "{\"data\":{\"config\":\"address-pools:\\n- name: default\\n  protocol: layer2\\n  addresses:\\n  - ${NET}.200-${NET}.210\\n\"}}"
kubectl -n metallb-system rollout restart deploy/controller ds/speaker

minikube image load heart-api:1.0.0
kubectl apply -f k8s/
kubectl -n heart rollout status deploy/heart-api
kubectl -n heart get pods,svc              # EXTERNAL-IP of service/heart-api
curl http://<EXTERNAL-IP>/health
curl -X POST http://<EXTERNAL-IP>/predict -H 'Content-Type: application/json' -d @api/sample_input.json
```

The EXTERNAL-IP depends on the machine's Minikube network; on the machine used for the report it
was `192.168.49.200`. `minikube start` resets the pool to an empty default, so run the three pool
commands again after every restart.

Some networks reset image pulls made from inside Minikube (`connection reset by peer`); Kubernetes
retries them. If a pod stays in `ImagePullBackOff`, pull the image on the host and load it:
`docker pull <image>` then `minikube image load <image>`. If `docker build` stalls while downloading
packages (common when the network's MTU is below Docker's 1500, e.g. on a VPN), build with
`docker build --network=host -t heart-api:1.0.0 .`.

The manifests are cloud-ready: on GKE, EKS or AKS the same `type: LoadBalancer` Service gets a
cloud load balancer, after pushing the image to a registry the cluster can pull from.

## 7. Monitoring and logging

```bash
kubectl apply -k monitoring/
python scripts/load_test.py http://<EXTERNAL-IP> 300
kubectl -n heart port-forward svc/prometheus 9090:9090   # http://localhost:9090/targets
kubectl -n heart port-forward svc/grafana 3000:3000      # http://localhost:3000
kubectl -n heart logs -l app=heart-api --tail=20         # JSON request logs
```
