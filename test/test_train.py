import json

import mlflow
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import heart.train as train
from heart.config import FEATURES, RANDOM_STATE, TARGET
from heart.models import make_pipeline


@pytest.fixture(scope="module")
def split(clean_df):
    X, y = clean_df[FEATURES], clean_df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def test_split_metrics_are_probabilities_and_complete(small_model, clean_df):
    m = train.split_metrics(small_model, clean_df[FEATURES], clean_df[TARGET])
    assert set(m) == set(train.SCORERS)
    assert all(0 <= v <= 1 for v in m.values())


@pytest.mark.parametrize(
    "clf, column",
    [(LogisticRegression(solver="liblinear", random_state=0), "coefficient"),
     (RandomForestClassifier(n_estimators=10, random_state=0), "importance")],
)  # fmt: skip
def test_feature_importance_is_sorted_by_magnitude(clf, column, split):
    X_train, _, y_train, _ = split
    table = train.feature_importance(make_pipeline(clf).fit(X_train, y_train))
    assert list(table.columns) == ["feature", column]
    assert len(table) == 22
    assert np.all(np.diff(table[column].abs().to_numpy()) <= 0)


def test_train_candidate_logs_parent_and_child_runs(split, tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test = split
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path / "reports")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp = mlflow.create_experiment("test", artifact_location=(tmp_path / "mlruns").as_uri())
    mlflow.set_experiment(experiment_id=exp)
    spec = {
        "estimator": LogisticRegression(solver="liblinear", random_state=0),
        "grid": {"model__C": [0.1, 1.0]},
    }
    context = {"tags": {"git_commit": "test"}, "data_card": {"rows": len(X_train)}}
    result = train.train_candidate("lr_test", spec, X_train, y_train, X_test, y_test, context)

    assert result["grid_size"] == 2
    assert 0 <= result["cv_f2_mean"] <= 1 and 0 <= result["test_roc_auc"] <= 1
    runs = mlflow.search_runs(experiment_ids=[exp])
    parent = runs[runs["run_id"] == result["run_id"]].iloc[0]
    children = runs[runs["tags.mlflow.parentRunId"] == result["run_id"]]
    assert len(children) == 2
    child = children.iloc[0]["run_id"]
    folds = mlflow.artifacts.load_dict(f"runs:/{child}/cv_folds.json")
    assert folds["parent_run"] == result["run_id"]
    assert all(len(v) == 5 for v in folds["fold_scores"].values())
    assert parent["params.refit_metric"] == "f2"
    assert {"metrics.cv_f2_mean", "metrics.test_recall"} <= set(runs.columns)
    signature = json.loads(
        mlflow.models.get_model_info(result["model_uri"]).signature.to_dict()["outputs"]
    )
    assert signature[0]["tensor-spec"]["shape"] == [-1, 2]  # probabilities of both classes
