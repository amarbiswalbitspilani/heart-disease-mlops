import numpy as np
import pandas as pd
import pytest

from heart.config import FEATURES, RAW_COLUMNS, TARGET
from heart.data import clean, load_clean, load_raw, save_clean, summarise


def test_raw_file_has_uci_shape():
    raw = load_raw()
    assert raw.shape == (303, 14)
    assert list(raw.columns) == RAW_COLUMNS


def test_question_marks_become_missing_values():
    raw = load_raw()
    assert raw["ca"].isna().sum() == 4
    assert raw["thal"].isna().sum() == 2
    assert raw.drop(columns=["ca", "thal"]).notna().all().all()


def test_clean_returns_features_and_binary_target(clean_df):
    assert list(clean_df.columns) == FEATURES + [TARGET]
    assert set(clean_df[TARGET].unique()) == {0, 1}
    assert all(np.issubdtype(t, np.number) for t in clean_df.dtypes)


def test_target_is_disease_grade_above_zero():
    raw = pd.DataFrame([[50, 1, 4, 130, 250, 0, 2, 140, 1, 1.0, 2, 0, 3, g] for g in range(5)],
                       columns=RAW_COLUMNS)  # fmt: skip
    assert clean(raw)[TARGET].tolist() == [0, 1, 1, 1, 1]


def test_clean_rejects_missing_column():
    with pytest.raises(ValueError, match="missing columns"):
        clean(load_raw().drop(columns=["chol"]))


def test_clean_rejects_unknown_disease_grade():
    raw = load_raw().head(3).copy()
    raw.loc[0, "num"] = 7
    with pytest.raises(ValueError, match="num"):
        clean(raw)


def test_save_and_load_round_trip(clean_df, tmp_path):
    path = save_clean(clean_df, tmp_path / "out" / "heart.csv")
    pd.testing.assert_frame_equal(load_clean(path), clean_df, check_dtype=False)


def test_summary_counts(clean_df):
    s = summarise(clean_df)
    assert s["rows"] == 303
    assert s["rows_with_missing"] == 6
    assert sum(s["class_counts"].values()) == 303
