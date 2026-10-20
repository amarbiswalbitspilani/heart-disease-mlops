"""Score one or more patients with the saved model.

Usage: python scripts/predict.py [path/to/input.json]   (default: api/sample_input.json)

The input is a JSON object (one patient) or a list of objects, in the same format as the
API request body (see api/schemas.py); fields are validated before scoring.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from api.schemas import PatientFeatures  # noqa: E402
from heart.config import FEATURES, MODEL_PATH  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "api" / "sample_input.json"
    records = json.loads(path.read_text())
    records = records if isinstance(records, list) else [records]
    model = joblib.load(MODEL_PATH)
    patients = [PatientFeatures(**r) for r in records]
    X = pd.DataFrame([p.to_features() for p in patients])[FEATURES].astype(float)
    proba = model.predict_proba(X)[:, 1]
    for record, p in zip(records, proba, strict=True):
        label = int(p >= 0.5)
        print(json.dumps({
            "input": record,
            "prediction": label,
            "label": "disease" if label else "no disease",
            "probability_of_disease": round(float(p), 4),
        }))  # fmt: skip


if __name__ == "__main__":
    main()
