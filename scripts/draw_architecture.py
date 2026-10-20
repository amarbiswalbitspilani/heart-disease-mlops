"""Draw the system architecture diagram to reports/figures/architecture.png.

Usage: python scripts/draw_architecture.py

Boxes carry the component names of a supervised-learning system (Orchestrator, Featurizer,
Model Builder, Evaluator, Model Server, Front-end, Performance Monitor, Ground-truth Collector,
Data Labeler, Databases) and the file, tool or command that plays each role in this project.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart.config import FIGURES_DIR  # noqa: E402

INK, MUTED, FUTURE, EDGE = "#0f172a", "#334155", "#94a3b8", "#475569"

# (bottom, top, title, fill colour)
LANES = [
    (10.10, 12.85, "Source control and CI (GitHub)", "#eef4fb"),
    (6.00, 9.85, "Data and training", "#f3f0fa"),
    (3.75, 5.75, "Packaging and deployment (manual)", "#fdf3e7"),
    (0.25, 3.55, "Serving and monitoring: Minikube, namespace heart", "#eef8f2"),
]

# (x, y, width, height, title, detail lines, style)
BOXES = [
    (0.25, 10.25, 2.75, 2.15, "GitHub repository",
     ["src/, api/, data/,", "notebooks/, test/,", "Dockerfile, k8s/,", "monitoring/, README,",
      ".github/workflows/"],
     "box"),
    (3.25, 10.25, 6.50, 2.15, "CI: GitHub Actions (push, pull request, manual)",
     ["artifacts per run: lint log, test results, model + metrics, docker logs"], "box"),
    (3.40, 10.72, 1.15, 0.50, "", ["lint (ruff)"], "chip"),
    (4.80, 10.72, 1.40, 0.50, "", ["unit tests", "(pytest)"], "chip"),
    (6.50, 11.05, 3.10, 0.50, "",
     ["docker job: build heart-api:ci,", "smoke test /health, /predict, 422"], "chip"),
    (6.50, 10.40, 3.10, 0.50, "",
     ["train job: download + heart.train,", "uploads model, metrics, MLflow runs"], "chip"),
    (0.25, 8.00, 3.00, 1.50, "Data store (Databases)",
     ["download_data.py (UCI id 45)", "clean: '?' -> NaN, num -> 0/1", "raw + clean CSV in Git,",
      "md5 tagged in every run"], "box"),
    (3.50, 8.00, 2.70, 1.50, "EDA notebook",
     ["notebooks/01_eda.ipynb", "class balance, histograms,", "correlations, age x sex"], "box"),
    (6.45, 8.00, 3.30, 1.50, "MLflow tracking + registry",
     ["runs: params, CV/train/test", "metrics, plots, model", "tags: git SHA, data md5",
      "registry alias 'champion'", "(lineage; not read by the API)"], "box"),
    (0.25, 6.15, 1.66, 1.40, "Load + split", ["stratified", "80/20:", "242 / 61 rows"], "box"),
    (2.21, 6.15, 1.66, 1.40, "Featurizer", ["impute, scale,", "one-hot", "13 -> 22 cols"], "box"),
    (4.17, 6.15, 1.66, 1.40, "Model Builder",
     ["GridSearchCV", "stratified", "5-fold CV", "LR, RF"], "box"),
    (6.13, 6.15, 1.66, 1.40, "Evaluator",
     ["selects by", "CV F2;", "test split:", "check + plots"], "box"),
    (8.09, 6.15, 1.66, 1.40, "Model file",
     ["models/", "model.joblib", "(selected", "model)"], "box"),
    (0.25, 3.90, 5.00, 1.40, "Container image heart-api:1.0.0",
     ["docker build -t heart-api:1.0.0 . (multi-stage)",
      "stage 1: runs heart.train -> model.joblib",
      "stage 2: python:3.11.17-slim, non-root user,",
      "src/heart/, api/, model.joblib; uvicorn :8000"],
     "box"),
    (5.65, 3.90, 4.10, 1.40, "Manual deployment (README)",
     ["minikube addons enable metallb + pool", "minikube image load heart-api:1.0.0",
      "kubectl apply -f k8s/: 2 pods, probes", "kubectl apply -k monitoring/"], "box"),
    (0.25, 1.85, 1.50, 1.35, "Client",
     ["curl or", "load_test.py", "POST /predict", "(JSON)"], "box"),
    (2.00, 1.85, 1.95, 1.35, "Load balancer",
     ["type LoadBalancer", "MetalLB pool IP,", "reference run:", "192.168.49.200:80"], "box"),
    (4.28, 1.90, 1.50, 1.05, "Front-end", ["validate -> 422", "label +", "confidence"], "chip"),
    (6.00, 1.90, 1.50, 1.05, "Model Server",
     ["Pipeline:", "Featurizer +", "selected model"], "chip"),
    (7.85, 1.85, 1.90, 1.35, "Monitor (ops)",
     ["Prometheus pulls", "/metrics every 5 s:", "requests, latency,", "class mix, confidence"],
     "box"),
    (4.20, 0.40, 3.40, 1.15, "Request logs (JSON, stdout)",
     ["id, path, status, latency_ms,", "prediction, confidence;",
      "no input features; kubectl logs"], "box"),
    (7.85, 0.40, 1.90, 1.15, "Grafana", ["dashboard,", "Prometheus", "data source"], "box"),
    (0.25, 0.40, 3.70, 1.20, "Future: Ground-truth Collector",
     ["confirmed diagnoses -> Data store,", "re-run Orchestrator; track accuracy.",
      "Data Labeler not needed (UCI labels)"], "future"),
]  # fmt: skip

# Frame around the steps python -m heart.train runs (download, cleaning and EDA run separately).
ORCH = (0.18, 6.05, 9.64, 1.85, "Orchestrator: python -m heart.train")

# Deployment frame around the two roles that share one FastAPI container in each pod.
POD = (4.20, 1.80, 3.40, 1.45, "heart-api: 2 pods, 1 FastAPI container each")

# (start, end, label, label position)
ARROWS = [
    ((3.00, 11.30), (3.25, 11.30), "", None),
    ((4.55, 10.97), (4.80, 10.97), "", None),
    ((6.20, 11.10), (6.50, 11.30), "", None),
    ((6.20, 10.85), (6.50, 10.65), "", None),
    ((8.05, 10.40), (8.05, 9.87), "", None),
    ((1.08, 8.00), (1.08, 7.55), "", None),
    ((3.25, 8.75), (3.50, 8.75), "", None),
    ((1.91, 6.85), (2.21, 6.85), "", None),
    ((3.87, 6.85), (4.17, 6.85), "", None),
    ((5.83, 6.85), (6.13, 6.85), "", None),
    ((7.79, 6.85), (8.09, 6.85), "", None),
    ((6.96, 7.55), (6.96, 8.00), "log run", (7.06, 7.78)),
    ((4.95, 6.05), (4.95, 5.30), "build stage 1 re-runs heart.train", (5.07, 5.87)),
    ((5.25, 4.60), (5.65, 4.60), "", None),
    ((6.60, 3.90), (6.60, 3.25), "creates", (6.70, 3.42)),
    ((1.75, 2.52), (2.00, 2.52), "", None),
    ((3.95, 2.52), (4.20, 2.52), "", None),
    ((5.78, 2.35), (6.00, 2.35), "", None),
    ((7.60, 2.52), (7.85, 2.52), "", None),
    ((5.90, 1.80), (5.90, 1.55), "", None),
    ((8.80, 1.85), (8.80, 1.55), "", None),
]


def rounded(ax, x, y, w, h, face, edge, style="-", width=1.1, radius=0.08) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=width, linestyle=style,
        )
    )  # fmt: skip


def draw() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 11.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.45)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for bottom, top, title, colour in LANES:
        rounded(ax, 0.1, bottom, 9.8, top - bottom, colour, "#c9d1db", width=1, radius=0.12)
        ax.text(0.25, top - 0.1, title, fontsize=11.5, fontweight="bold", color=MUTED, va="top")

    x, y, w, h, title = ORCH
    rounded(ax, x, y, w, h, "#e7e1f6", "#8b7fc0", radius=0.1)
    ax.text(4.0, y + h - 0.15, title, ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=MUTED)  # fmt: skip

    x, y, w, h, title = POD
    rounded(ax, x, y, w, h, "#d5ecdd", "#6b9f7e", radius=0.1)
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="center", fontsize=9, color=MUTED)

    for x, y, w, h, title, lines, style in BOXES:
        future, chip = style == "future", style == "chip"
        colour = FUTURE if future else INK
        rounded(ax, x, y, w, h, "white", FUTURE if future else EDGE, "--" if future else "-")
        size = 9 if chip else 10
        if title:
            ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center",
                    fontsize=10.5 if chip else 11, fontweight="bold", color=colour)  # fmt: skip
            ax.text(x + w / 2, y + h - 0.4, "\n".join(lines), ha="center", va="top",
                    fontsize=size, color=colour, linespacing=1.3)  # fmt: skip
        else:
            ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
                    fontsize=size, color=colour, linespacing=1.2)  # fmt: skip

    for start, end, label, at in ARROWS:
        style = {"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.3, "shrinkA": 0,
                 "shrinkB": 0, "mutation_scale": 11}  # fmt: skip
        ax.annotate("", xy=end, xytext=start, arrowprops=style)
        if label:
            ax.text(*at, label, ha="left", va="center", fontsize=9, color=MUTED)

    ax.text(5, 13.15, "Heart disease risk prediction: MLOps architecture", ha="center",
            va="center", fontsize=14, fontweight="bold", color=INK)  # fmt: skip
    return fig


def main() -> None:
    fig = draw()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
