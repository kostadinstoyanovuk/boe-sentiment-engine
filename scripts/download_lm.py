"""
Download the Loughran-McDonald Master Dictionary.

The LM master dictionary is hosted by the Software Repository for
Accounting and Finance (SRAF) at Notre Dame. This script downloads
the CSV and saves it to data/lm_master.csv.

If automatic download fails, download manually from:
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/

Save the CSV as data/lm_master.csv in the project root.
"""

import csv
import sys
from pathlib import Path

import requests

SRAF_URL = (
    "https://sraf.nd.edu/wp-content/uploads/2024/06/"
    "Loughran-McDonald_MasterDictionary_1993-2023.csv"
)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "lm_master.csv"


def download():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Already exists: {OUTPUT_PATH}")
        return

    print(f"Downloading LM master dictionary from SRAF...")
    try:
        resp = requests.get(SRAF_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        print(f"Download manually from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")
        print(f"Save the CSV as: {OUTPUT_PATH}")
        sys.exit(1)

    OUTPUT_PATH.write_bytes(resp.content)

    # Verify the file looks right
    with open(OUTPUT_PATH, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "Word" not in headers or "Negative" not in headers:
            print(f"Warning: CSV headers look unexpected: {headers[:5]}")
            print("Expected 'Word', 'Negative', 'Positive' columns.")
            sys.exit(1)
        row_count = sum(1 for _ in reader)

    print(f"Saved: {OUTPUT_PATH} ({row_count} entries)")


if __name__ == "__main__":
    download()
