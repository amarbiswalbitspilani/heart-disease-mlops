"""Send a stream of prediction requests to the API, to exercise monitoring.

Usage: python scripts/load_test.py [BASE_URL] [N]     (default: http://localhost:8000 300)

Patients are drawn from the cleaned dataset and sent in the API's request format; about
5% of requests carry an invalid value, so the error-rate panels have something to show.
"""

import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from api.schemas import FIELD_TO_COLUMN  # noqa: E402
from heart.data import load_clean  # noqa: E402

INT_FIELDS = {"age", "sex", "chest_pain_type", "fasting_blood_sugar_gt_120", "resting_ecg",
              "exercise_induced_angina", "st_slope", "major_vessels", "thalassemia"}  # fmt: skip


def to_request(row) -> dict:
    body = {}
    for field, column in FIELD_TO_COLUMN.items():
        value = row[column]
        if value != value:  # NaN -> unknown
            body[field] = None
        else:
            body[field] = int(value) if field in INT_FIELDS else float(value)
    return body


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def main() -> None:
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rng = random.Random(7)
    rows = load_clean().to_dict("records")
    statuses = Counter()
    start = time.perf_counter()
    for i in range(n):
        body = to_request(rng.choice(rows))
        if rng.random() < 0.05:
            body["age"] = 150  # out of range -> 422
        statuses[post(f"{base}/predict", body)] += 1
        time.sleep(rng.uniform(0.01, 0.08))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{n} sent")
    elapsed = time.perf_counter() - start
    by_status = dict(sorted(statuses.items()))
    print(f"Sent {n} requests in {elapsed:.1f}s; responses by status: {by_status}")


if __name__ == "__main__":
    main()
