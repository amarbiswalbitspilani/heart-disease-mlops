"""Feature preprocessing as a scikit-learn ColumnTransformer.

The transformer is the first step of every model Pipeline, so imputation, scaling and
one-hot encoding are fitted on training folds only and ship inside the saved model.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart.config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("binary", binary, BINARY_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
