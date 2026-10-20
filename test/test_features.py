import numpy as np

from heart.config import FEATURES
from heart.features import build_preprocessor


def test_preprocessor_output_has_no_missing_values(clean_df):
    X = clean_df[FEATURES]
    assert X.isna().any().any()
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_one_hot_encoding_expands_categorical_columns(clean_df):
    pre = build_preprocessor().fit(clean_df[FEATURES])
    names = list(pre.get_feature_names_out())
    assert len(names) == 22
    assert {"cp_1.0", "cp_4.0", "thal_7.0", "slope_3.0"} <= set(names)


def test_numeric_columns_are_standardised(clean_df):
    pre = build_preprocessor().fit(clean_df[FEATURES])
    out = pre.transform(clean_df[FEATURES])
    age = out[:, list(pre.get_feature_names_out()).index("age")]
    assert abs(age.mean()) < 1e-9 and abs(age.std() - 1) < 1e-9


def test_unseen_category_does_not_fail(clean_df):
    pre = build_preprocessor().fit(clean_df[FEATURES])
    row = clean_df[FEATURES].head(1).copy()
    row["cp"] = 9
    assert pre.transform(row).shape == (1, 22)
