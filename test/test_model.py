import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from heart.config import FEATURES, TARGET
from heart.models import CANDIDATES, make_pipeline


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_candidate_grid_parameters_exist(name):
    spec = CANDIDATES[name]
    params = make_pipeline(spec["estimator"]).get_params()
    assert set(spec["grid"]) <= set(params)


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_candidate_trains_and_predicts_probabilities(name, clean_df):
    X, y = clean_df[FEATURES], clean_df[TARGET]
    model = make_pipeline(clone(CANDIDATES[name]["estimator"])).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(model.predict(X)) <= {0, 1}


def test_model_beats_majority_class_on_held_out_rows(clean_df):
    X, y = clean_df[FEATURES], clean_df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    model = make_pipeline(clone(CANDIDATES["logistic_regression"]["estimator"])).fit(X_tr, y_tr)
    majority = max(y_te.mean(), 1 - y_te.mean())
    assert (model.predict(X_te) == y_te).mean() > majority


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_training_is_reproducible(name, clean_df):
    X, y = clean_df[FEATURES], clean_df[TARGET]
    # L1 is the liblinear case that shuffles the data, so it is the one a missing seed would break.
    est = clone(CANDIDATES[name]["estimator"])
    if name == "logistic_regression":
        est.set_params(l1_ratio=1.0, class_weight="balanced")
    a = make_pipeline(clone(est)).fit(X, y).predict_proba(X)
    b = make_pipeline(clone(est)).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


def test_parity_check_accepts_the_same_model_and_rejects_another(model_file, clean_df, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_model_parity.py"
    X, y = clean_df[FEATURES], clean_df[TARGET]
    other = make_pipeline(clone(CANDIDATES["logistic_regression"]["estimator"]).set_params(C=0.01))
    joblib.dump(other.fit(X, y), tmp_path / "other.joblib")
    same = subprocess.run([sys.executable, script, model_file, model_file], capture_output=True)
    differ = subprocess.run([sys.executable, script, model_file, tmp_path / "other.joblib"],
                            capture_output=True)  # fmt: skip
    assert same.returncode == 0
    assert differ.returncode != 0
