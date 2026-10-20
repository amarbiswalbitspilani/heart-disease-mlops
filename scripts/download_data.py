"""Download the UCI Heart Disease (Cleveland) file and write the cleaned CSV.

Usage: python scripts/download_data.py

The raw file is fetched from the UCI Machine Learning Repository and checked against the MD5 of
the reference copy kept in the repository; if UCI is unreachable or serves different content,
the script warns and continues from the committed copy.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart.config import DATA_CLEAN, DATA_RAW, RAW_MD5, UCI_URL  # noqa: E402
from heart.data import clean, load_raw, save_clean, summarise  # noqa: E402


def download(url: str = UCI_URL, dest: Path = DATA_RAW) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read()
        digest = hashlib.md5(body).hexdigest()
        if digest != RAW_MD5:
            raise ValueError(f"content differs from the reference file (md5 {digest})")
        dest.write_bytes(body)
        print(f"Downloaded {url} -> {dest.relative_to(DATA_RAW.parents[2])}")
    except Exception as exc:  # network down, UCI moved, proxy...
        if not dest.exists():
            raise SystemExit(f"Download failed and no local copy exists: {exc}") from exc
        print(f"WARNING: download failed ({exc}); using the committed copy {dest.name}")


def main() -> None:
    download()
    df = clean(load_raw())
    save_clean(df)
    print(f"Wrote {DATA_CLEAN.relative_to(DATA_RAW.parents[2])}")
    print(json.dumps(summarise(df), indent=2))


if __name__ == "__main__":
    main()
