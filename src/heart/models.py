"""Candidate models and their tuning grids. Each model is a Pipeline: preprocessing + classifier."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from heart.config import RANDOM_STATE
from heart.features import build_preprocessor


def make_pipeline(classifier) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", classifier)])


# Logistic regression is the simple, interpretable baseline; the random forest tests whether
# non-linear interactions (e.g. age x sex, chest pain x exercise angina) add signal.
CANDIDATES = {
    "logistic_regression": {
        # liblinear shuffles the data internally, so it needs a seed to be reproducible.
        "estimator": LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=RANDOM_STATE
        ),
        "grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            # l1_ratio 0 = L2 (ridge) penalty, 1 = L1 (lasso) penalty, which can drop features.
            "model__l1_ratio": [0.0, 1.0],
            "model__class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"],
        },
    },
}
