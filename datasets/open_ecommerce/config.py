"""Configuration for Open E-Commerce 1.0 dataset (Harvard Dataverse)."""

from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent.parent / "cache"
OUTPUT_DIR = Path(__file__).parent / "output"

# Dataset info
DATASET_DOI = "doi:10.7910/DVN/YGLYDY"
DATASET_URL = "https://dataverse.harvard.edu"
DATA_FILE = DATA_DIR / "amazon-purchases.csv"

# Filtering thresholds
MIN_OBSERVATIONS = 5  # Minimum purchase periods per user (lowered for category-level analysis)
MIN_PRICE = 0.01  # Minimum price per item ($)
MAX_PRICE = 1000.0  # Maximum price per item ($)

# Time aggregation
TIME_PERIOD = "month"  # Aggregate to monthly periods

# Top categories (by volume) - Amazon product categories
TOP_N_CATEGORIES = 50  # Number of categories to include (use raw categories, not groups)

# Analysis thresholds (for tiered reporting)
HIGH_RATIONALITY_THRESHOLD = 0.95  # AEI >= 0.95 is "highly rational"
LOW_RATIONALITY_THRESHOLD = 0.70   # AEI < 0.70 is "erratic"

# Performance targets
MAX_PROCESSING_TIME_SECONDS = 300  # 5 minute target for full analysis
GARP_TOLERANCE = 1e-10             # Numerical tolerance for GARP checks
AEI_TOLERANCE = 1e-4               # Tolerance for AEI binary search (looser for speed)

# Category mapping - common Amazon categories to aggregate
CATEGORY_GROUPS = {
    "Electronics": ["Electronics", "Computers", "Cell Phones", "Camera & Photo"],
    "Home": ["Home & Kitchen", "Kitchen & Dining", "Furniture", "Bedding"],
    "Books": ["Books", "Kindle Store", "Audible Books"],
    "Clothing": ["Clothing, Shoes & Jewelry", "Women", "Men", "Shoes"],
    "Health": ["Health & Household", "Beauty & Personal Care", "Personal Care"],
    "Grocery": ["Grocery & Gourmet Food", "Grocery"],
    "Toys": ["Toys & Games", "Baby"],
    "Sports": ["Sports & Outdoors", "Outdoors"],
    "Office": ["Office Products", "Industrial & Scientific"],
    "Media": ["Movies & TV", "Music", "Video Games"],
}
