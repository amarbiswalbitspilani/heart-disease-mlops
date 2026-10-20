"""Check that two saved models give the same predictions on the cleaned dataset.

Usage: python scripts/check_model_parity.py MODEL_A MODEL_B [DATA_CSV]

The bytes of a saved model can differ between machines, because the linear-algebra library
picks CPU-specific kernels; reproducibility is therefore checked on the predictions.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart.config import DATA_CLEAN, FEATURES  # noqa: E402

TOLERANCE = 1e-9


def main() -> None:
    # Both files are models produced by this project's own training (joblib runs code on load).
    a, b = (joblib.load(p) for p in sys.argv[1:3])
    X = pd.read_csv(Path(sys.argv[3]) if len(sys.argv) > 3 else DATA_CLEAN)[FEATURES]
    diff = float(np.abs(a.predict_proba(X) - b.predict_proba(X)).max())
    same_labels = bool((a.predict(X) == b.predict(X)).all())
    print(
        f"{len(X)} rows: max |probability difference| = {diff:.1e}; identical labels: {same_labels}"
    )
    if diff > TOLERANCE or not same_labels:
        raise SystemExit("the models disagree")


if __name__ == "__main__":
    main()
