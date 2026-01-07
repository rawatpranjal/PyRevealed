"""Download Open E-Commerce 1.0 dataset from Harvard Dataverse.

This dataset contains Amazon purchase history for 5,027 US consumers (2018-2022).
Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY

Reference:
    Goldfarb, A., Tucker, C., & Wang, Y. (2024). Open e-commerce 1.0, five years
    of crowdsourced U.S. Amazon purchase histories with user demographics.
    Scientific Data, 11(527).
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DATA_FILE, DATASET_DOI, DATASET_URL


def get_dataset_files() -> list[dict]:
    """
    Get list of files in the dataset via Dataverse API.

    Returns:
        List of file metadata dicts with 'id', 'filename', 'contentType', etc.
    """
    api_url = f"{DATASET_URL}/api/datasets/:persistentId?persistentId={DATASET_DOI}"

    print(f"  Fetching dataset metadata from: {api_url}")
    response = requests.get(api_url)
    response.raise_for_status()

    data = response.json()
    files = data.get("data", {}).get("latestVersion", {}).get("files", [])

    return files


def download_file(file_id: int, filename: str, output_path: Path) -> Path:
    """
    Download a single file from Dataverse.

    Args:
        file_id: Dataverse file ID
        filename: Original filename
        output_path: Path to save the file

    Returns:
        Path to downloaded file
    """
    download_url = f"{DATASET_URL}/api/access/datafile/{file_id}"

    print(f"  Downloading {filename}...")
    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    # Get file size if available
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = 100 * downloaded / total_size
                print(f"\r    Progress: {pct:.1f}%", end="", flush=True)

    if total_size > 0:
        print()  # newline after progress

    return output_path


def download_dataset() -> Path:
    """
    Download the Open E-Commerce dataset.

    Returns:
        Path to the main data file (amazon-purchases.csv)
    """
    print("Downloading Open E-Commerce 1.0 dataset...")
    print("  Source: Harvard Dataverse")
    print(f"  DOI: {DATASET_DOI}")

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Get file list
    files = get_dataset_files()

    if not files:
        raise RuntimeError("No files found in dataset. API may have changed.")

    print(f"  Found {len(files)} files in dataset")

    # Find and download the main data file
    main_file = None
    for f in files:
        datafile = f.get("dataFile", {})
        filename = datafile.get("filename", "")

        if "amazon-purchases" in filename.lower() or filename.endswith(".csv"):
            file_id = datafile.get("id")
            print(f"  Found main data file: {filename} (id={file_id})")
            main_file = f
            break

    if main_file is None:
        # List available files
        print("  Available files:")
        for f in files:
            print(f"    - {f.get('dataFile', {}).get('filename', 'unknown')}")
        raise RuntimeError("Could not find amazon-purchases.csv in dataset")

    # Download the file
    file_id = main_file["dataFile"]["id"]
    filename = main_file["dataFile"]["filename"]
    output_path = DATA_DIR / filename

    download_file(file_id, filename, output_path)

    # Rename to standard name if different
    if output_path != DATA_FILE and output_path.suffix == ".csv":
        output_path.rename(DATA_FILE)
        output_path = DATA_FILE

    print(f"  Saved to: {output_path}")

    # Also download survey data if available
    for f in files:
        datafile = f.get("dataFile", {})
        filename = datafile.get("filename", "")
        if "survey" in filename.lower():
            file_id = datafile.get("id")
            survey_path = DATA_DIR / filename
            if not survey_path.exists():
                download_file(file_id, filename, survey_path)

    return output_path


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
