"""Download the UCI Bank Marketing dataset and extract bank-full.csv.

Source: UCI Machine Learning Repository, dataset 222 (Bank Marketing)
https://archive.ics.uci.edu/dataset/222/bank+marketing

The UCI zip contains two nested archives (bank.zip and bank-additional.zip);
bank-full.csv lives inside bank.zip. This script downloads the outer zip,
extracts bank-full.csv next to this script, and sanity-checks the result.

Usage:
    python data/download_data.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUT_PATH = Path(__file__).resolve().parent / "bank-full.csv"


def main() -> None:
    print(f"Downloading {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL) as resp:
        outer = resp.read()

    with zipfile.ZipFile(io.BytesIO(outer)) as z_outer:
        bank_zip_bytes = z_outer.read("bank.zip")

    with zipfile.ZipFile(io.BytesIO(bank_zip_bytes)) as z_bank:
        csv_bytes = z_bank.read("bank-full.csv")

    OUT_PATH.write_bytes(csv_bytes)

    lines = csv_bytes.decode("utf-8").strip().splitlines()
    header = lines[0]
    n_rows = len(lines) - 1
    print(f"Wrote {OUT_PATH} ({len(csv_bytes):,} bytes)")
    print(f"Header: {header}")
    print(f"Rows: {n_rows:,}")
    assert n_rows == 45_211, f"Unexpected row count: {n_rows}"
    # The CSV ships with a quoted header ("age";"job";...), so compare unquoted.
    assert header.replace('"', "").startswith("age;job;marital"), "Unexpected header"
    print("Sanity check passed: 45,211 rows.")


if __name__ == "__main__":
    main()
