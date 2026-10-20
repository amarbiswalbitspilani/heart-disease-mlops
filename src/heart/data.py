"""Load the raw UCI Cleveland file and turn it into a typed, binary-target table.

Cleaning is deliberately limited to what does not learn from the data: marking the
"?" placeholders as missing, setting numeric types and binarising the target.
Imputation, scaling and encoding are fitted inside the model pipeline (features.py),
so no statistic from the test rows reaches the training step.
"""

from pathlib import Path

import pandas as pd

from heart.config import DATA_CLEAN, DATA_RAW, FEATURES, RAW_COLUMNS, TARGET


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """Read the comma-separated UCI file, which has no header and uses '?' for missing."""
    return pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the 13 features plus a binary target (0 = no disease, 1 = disease)."""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    df = raw[RAW_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    if not df["num"].dropna().isin([0, 1, 2, 3, 4]).all():
        raise ValueError("num must take values 0-4")
    # UCI grades disease severity 0-4; any value above 0 means disease is present.
    df[TARGET] = (df["num"] > 0).astype(int)
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: Path = DATA_CLEAN) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path = DATA_CLEAN) -> pd.DataFrame:
    return pd.read_csv(path)


def summarise(df: pd.DataFrame) -> dict:
    """Counts used in the README and report: rows, missing values, duplicates, class balance."""
    return {
        "rows": int(len(df)),
        "features": len(FEATURES),
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n},
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {int(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
    }
