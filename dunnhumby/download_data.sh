#!/bin/bash
# Download Dunnhumby "The Complete Journey" dataset from Kaggle
# Requires: kaggle CLI installed and authenticated
#
# Setup:
#   pip install kaggle
#   # Create ~/.kaggle/kaggle.json with your API credentials
#   # See: https://www.kaggle.com/docs/api
#
# Usage:
#   cd dunnhumby && ./download_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

echo "=============================================="
echo " Dunnhumby Dataset Downloader"
echo "=============================================="

# Check if kaggle CLI is available
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found."
    echo "Install with: pip install kaggle"
    echo "Then configure credentials: https://www.kaggle.com/docs/api"
    exit 1
fi

# Create data directory
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo ""
echo "Downloading Dunnhumby 'The Complete Journey' dataset..."
echo "Dataset: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey"
echo ""

kaggle datasets download -d frtgnn/dunnhumby-the-complete-journey

echo ""
echo "Extracting files..."
unzip -o dunnhumby-the-complete-journey.zip

echo ""
echo "Cleaning up..."
rm -f dunnhumby-the-complete-journey.zip

echo ""
echo "=============================================="
echo " Download complete!"
echo "=============================================="
echo ""
echo "Data directory: ${DATA_DIR}"
echo ""
ls -la "${DATA_DIR}"
