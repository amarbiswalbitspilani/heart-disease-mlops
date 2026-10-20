"""Paths, column definitions and constants shared by every stage of the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "processed.cleveland.data"
DATA_CLEAN = ROOT / "data" / "processed" / "heart_clean.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# MD5 of the reference copy committed in data/raw; a download with other content is not used.
RAW_MD5 = "2d91a8ff69cfd9616aa47b59d6f843db"

# Column order of processed.cleveland.data, as documented by UCI (dataset id 45).
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
