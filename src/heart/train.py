"""Tune, evaluate and track every candidate model, then save the selected one.

Usage: python -m heart.train      (run from the repository root with src/ on PYTHONPATH)

For each candidate: grid search with stratified 5-fold cross-validation on the training
split (refit on F2), cross-validated metrics as mean and standard deviation, metrics on
the training split and on the held-out test split, and one MLflow run carrying parameters,
metrics, tags, plots, tables and the model itself.

The model with the best cross-validated F2 score is selected: F2 weights recall twice as
much as precision, because a missed disease case (false negative) is the costly error.
Selection uses cross-validation only, so the test split stays an unbiased final check.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from heart.config import (  # noqa: E402
    CV_FOLDS,
    DATA_CLEAN,
    DATA_RAW,
    FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    ROOT,
    TARGET,
    TEST_SIZE,
    UCI_URL,
)
from heart.data import load_clean  # noqa: E402
from heart.models import CANDIDATES, make_pipeline  # noqa: E402

EXPERIMENT = "heart-disease-classification"
REGISTERED_MODEL = "heart-disease-classifier"
SCORERS = {
    "accuracy": "accuracy",
    # The strongest L1 penalty can zero every coefficient and predict no positives in a fold;
    # precision is then scored 0 instead of raising a warning.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "f2": make_scorer(fbeta_score, beta=2),
    "roc_auc": "roc_auc",
}
# One metric for tuning and for choosing between models: F2, recall-weighted. ROC-AUC breaks ties.
REFIT_METRIC = "f2"
SELECTION_METRIC = "f2"
TIE_BREAK_METRIC = "roc_auc"
# Types the skops loader must accept; the files are produced by this pipeline itself.
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def split_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "f2": fbeta_score(y, pred, beta=2),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_plots(name: str, model, X_test, y_test, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, display in [
        ("confusion_matrix", ConfusionMatrixDisplay),
        ("roc_curve", RocCurveDisplay),
        ("pr_curve", PrecisionRecallDisplay),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        if kind == "confusion_matrix":
            display.from_estimator(
                model, X_test, y_test, display_labels=["No disease", "Disease"],
                cmap="Blues", colorbar=False, ax=ax,
            )  # fmt: skip
        else:
            display.from_estimator(model, X_test, y_test, name=name, ax=ax)
        ax.set_title(f"{name}: {kind.replace('_', ' ')} (test split)")
        fig.tight_layout()
        path = out_dir / f"{name}_{kind}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def feature_importance(model) -> pd.DataFrame | None:
    clf = model.named_steps["model"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values, kind = clf.feature_importances_, "importance"
    elif hasattr(clf, "coef_"):
        values, kind = clf.coef_[0], "coefficient"
    else:
        return None
    table = pd.DataFrame({"feature": names, kind: values})
    return table.reindex(table[kind].abs().sort_values(ascending=False).index)


def train_candidate(name, spec, X_train, y_train, X_test, y_test, context) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        make_pipeline(spec["estimator"]), spec["grid"], scoring=SCORERS,
        refit=REFIT_METRIC, cv=cv, n_jobs=-1, return_train_score=False,
    )  # fmt: skip
    search.fit(X_train, y_train)
    best = search.best_estimator_
    i = search.best_index_
    cv_res = search.cv_results_
    cv_metrics = {}
    for m in SCORERS:
        cv_metrics[f"cv_{m}_mean"] = float(cv_res[f"mean_test_{m}"][i])
        cv_metrics[f"cv_{m}_std"] = float(cv_res[f"std_test_{m}"][i])
    train_m = {f"train_{k}": v for k, v in split_metrics(best, X_train, y_train).items()}
    test_m = {f"test_{k}": v for k, v in split_metrics(best, X_test, y_test).items()}

    run_dir = REPORTS_DIR / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    grid_table = pd.DataFrame(cv_res).drop(columns=["params"])
    grid_table.to_csv(run_dir / "cv_results.csv", index=False)
    preds = X_test.copy()
    preds["actual"] = y_test.values
    preds["predicted"] = best.predict(X_test)
    preds["probability_of_disease"] = best.predict_proba(X_test)[:, 1].round(4)
    preds.to_csv(run_dir / "test_predictions.csv", index=False)
    importance = feature_importance(best)
    if importance is not None:
        importance.to_csv(run_dir / "feature_importance.csv", index=False)
    plots = save_plots(name, best, X_test, y_test, run_dir)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, **context["tags"]})
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_params(
            {
                "grid_size": len(cv_res["params"]),
                "cv_folds": CV_FOLDS,
                "refit_metric": REFIT_METRIC,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "imputer_numeric": "median",
                "imputer_categorical": "most_frequent",
                "scaler": "StandardScaler",
                "encoder": "OneHotEncoder(handle_unknown=ignore)",
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )
        mlflow.log_metrics({**cv_metrics, **train_m, **test_m})
        for p in plots:
            mlflow.log_artifact(str(p), artifact_path="plots")
        for f in ["cv_results.csv", "test_predictions.csv", "feature_importance.csv"]:
            if (run_dir / f).exists():
                mlflow.log_artifact(str(run_dir / f), artifact_path="tables")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="environment")
        if (ROOT / "Dockerfile").exists():
            mlflow.log_artifact(str(ROOT / "Dockerfile"), artifact_path="environment")
        mlflow.log_dict(context["data_card"], "data/data_card.json")
        mlflow.log_text(X_train.head().to_csv(index=False), "data/train_head.csv")
        example = X_train.head(3)
        model_info = mlflow.sklearn.log_model(
            best, name="model", input_example=example,
            signature=infer_signature(example, best.predict_proba(example)),
            pyfunc_predict_fn="predict_proba",
            # skops format: loading runs no arbitrary code; only these reviewed types are allowed.
            serialization_format="skops", skops_trusted_types=SKOPS_TRUSTED,
        )  # fmt: skip
        # Every grid combination as a child run: its parameters and cross-validated scores.
        for k, params in enumerate(cv_res["params"]):
            with mlflow.start_run(run_name=f"{name}-grid-{k:02d}", nested=True):
                mlflow.set_tag(
                    "grid_rank_" + REFIT_METRIC, int(cv_res[f"rank_test_{REFIT_METRIC}"][k])
                )
                mlflow.log_params({p.replace("model__", ""): v for p, v in params.items()})
                mlflow.log_metrics(
                    {f"cv_{m}_{stat}": float(cv_res[f"{stat}_test_{m}"][k])
                     for m in SCORERS for stat in ("mean", "std")}
                )  # fmt: skip
                folds = {m: [float(cv_res[f"split{f}_test_{m}"][k]) for f in range(CV_FOLDS)]
                         for m in SCORERS}  # fmt: skip
                mlflow.log_dict({"parent_run": run.info.run_id, "fold_scores": folds},
                                "cv_folds.json")  # fmt: skip
        run_id = run.info.run_id

    return {
        "name": name,
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "grid_size": len(cv_res["params"]),
        "estimator": best,
        **cv_metrics,
        **train_m,
        **test_m,
    }


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    df = load_clean(DATA_CLEAN)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    overlap = set(X_train.index) & set(X_test.index)
    if overlap or df.duplicated().any():
        raise RuntimeError("train/test overlap or duplicate rows in the cleaned data")

    context = {
        "tags": {
            "git_commit": git_commit(),
            "data_md5_raw": md5(DATA_RAW),
            "data_md5_clean": md5(DATA_CLEAN),
            "requirements_md5": md5(ROOT / "requirements.txt"),
        },
        "data_card": {
            "dataset": "Heart Disease, UCI Machine Learning Repository (id 45), Cleveland",
            "source_url": UCI_URL,
            "rows": int(len(df)),
            "features": FEATURES,
            "target": "num > 0 -> 1 (disease present), num == 0 -> 0",
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "missing_values": {c: int(n) for c, n in X.isna().sum().items() if n},
        },
    }

    results = [
        train_candidate(name, spec, X_train, y_train, X_test, y_test, context)
        for name, spec in CANDIDATES.items()
    ]
    winner = max(
        results, key=lambda r: (r[f"cv_{SELECTION_METRIC}_mean"], r[f"cv_{TIE_BREAK_METRIC}_mean"])
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner["estimator"], MODEL_PATH)
    version = mlflow.register_model(winner["model_uri"], REGISTERED_MODEL)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "selected_by", "cv_f2_mean")

    summary = {
        "selected_model": winner["name"],
        "selection_rule": (
            f"highest cv_{SELECTION_METRIC}_mean (tie-break cv_{TIE_BREAK_METRIC}_mean)"
        ),
        "registered_model": {"name": REGISTERED_MODEL, "version": int(version.version)},
        "split": {"train": len(X_train), "test": len(X_test)},
        "models": [{k: v for k, v in r.items() if k != "estimator"} for r in results],
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    keep = [c for c in runs.columns if c.startswith(("run_id", "tags.mlflow.runName", "start_time",
                                                     "params.", "metrics."))]  # fmt: skip
    head = ["run_id", "tags.mlflow.runName", "tags.mlflow.parentRunId", "start_time"]
    runs = runs.reindex(columns=list(dict.fromkeys([*runs.columns, "tags.mlflow.parentRunId"])))
    runs[head + sorted(c for c in keep if c not in head)].to_csv(
        REPORTS_DIR / "mlflow_runs.csv", index=False
    )

    rows = []
    for r in results:
        rows.append(
            {"model": r["name"], **{m: f"{r[f'cv_{m}_mean']:.3f} ± {r[f'cv_{m}_std']:.3f}"
                                    for m in SCORERS}}
        )  # fmt: skip
    print("Cross-validation (training split, mean ± std over 5 folds):")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nHeld-out test split:")
    print(pd.DataFrame([{"model": r["name"], **{m: round(r[f"test_{m}"], 3) for m in SCORERS}}
                        for r in results]).to_string(index=False))  # fmt: skip
    print(f"\nSelected: {winner['name']} -> {MODEL_PATH.relative_to(ROOT)}, "
          f"registered as {REGISTERED_MODEL} v{version.version} (alias 'champion')")  # fmt: skip


if __name__ == "__main__":
    main()
