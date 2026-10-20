import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from heart.config import FEATURES, TARGET
from heart.data import clean, load_raw
from heart.models import make_pipeline


@pytest.fixture(scope="session")
def clean_df():
    return clean(load_raw())


@pytest.fixture(scope="session")
def small_model(clean_df):
    model = make_pipeline(LogisticRegression(solver="liblinear", max_iter=500, random_state=0))
    return model.fit(clean_df[FEATURES], clean_df[TARGET])


@pytest.fixture(scope="session")
def model_file(small_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(small_model, path)
    return path
