"""Configuration for UCI Online Retail dataset."""

from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent.parent / "cache"

# Dataset info
DATASET_ID = 352  # UCI ML Repository ID
DATA_FILE = DATA_DIR / "online_retail.xlsx"

# Filtering thresholds
MIN_TRANSACTIONS = 5  # Minimum transactions per customer
MIN_UNIT_PRICE = 0.01  # Minimum price per unit (GBP)
MAX_UNIT_PRICE = 500.0  # Maximum price per unit (GBP)

# Time aggregation
TIME_PERIOD = "month"  # Aggregate to monthly periods

# Top product categories (by volume) - will be computed from data
# Using StockCode prefix patterns for grouping
TOP_N_PRODUCTS = 50  # Number of distinct products to include
