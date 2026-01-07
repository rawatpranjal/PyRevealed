"""Download UCI Online Retail dataset.

This dataset contains transactional data from a UK-based online retail store.
Source: https://archive.ics.uci.edu/dataset/352/online+retail

Columns:
- InvoiceNo: Invoice number (6-digit, prefix 'C' = cancellation)
- StockCode: Product code (5-digit)
- Description: Product name
- Quantity: Quantity per transaction
- InvoiceDate: Invoice date and time
- UnitPrice: Price per unit (GBP)
- CustomerID: Customer number (5-digit)
- Country: Country name
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DATA_FILE

# Direct download URL for the dataset
DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"


def download_via_curl(url: str, output_path: Path) -> bool:
    """Download file using curl (handles SSL better on macOS)."""
    print(f"  Downloading from: {url}")
    result = subprocess.run(
        ["curl", "-L", "-o", str(output_path), url],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def download_via_ucimlrepo() -> bool:
    """Try to download using ucimlrepo package."""
    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd

        print("  Trying ucimlrepo package...")
        online_retail = fetch_ucirepo(id=352)
        df = online_retail.data.original

        print(f"  Downloaded {len(df):,} transactions")
        df.to_excel(DATA_FILE, index=False)
        return True
    except Exception as e:
        print(f"  ucimlrepo failed: {e}")
        return False


def download_dataset() -> Path:
    """
    Download the UCI Online Retail dataset.

    Returns:
        Path to the downloaded Excel file
    """
    print("Downloading UCI Online Retail dataset...")
    print("  Source: UCI Machine Learning Repository (ID: 352)")

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try ucimlrepo first, then fall back to curl
    if not download_via_ucimlrepo():
        print("  Falling back to direct download via curl...")
        if not download_via_curl(DOWNLOAD_URL, DATA_FILE):
            raise RuntimeError("Failed to download dataset")

    # Verify the file exists and has content
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {DATA_FILE} is missing or empty")

    print(f"  Saved to: {DATA_FILE}")

    # Load to verify and show stats
    import pandas as pd
    df = pd.read_excel(DATA_FILE)
    print(f"  Transactions: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    return DATA_FILE


def main():
    """Download the dataset."""
    if DATA_FILE.exists():
        print(f"Dataset already exists: {DATA_FILE}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != "y":
            print("Skipping download.")
            return

    download_dataset()
    print("\nDone! Run data_loader.py to preprocess the data.")


if __name__ == "__main__":
    main()
