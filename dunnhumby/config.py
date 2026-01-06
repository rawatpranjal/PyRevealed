"""Configuration constants for Dunnhumby analysis."""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directories
DUNNHUMBY_DIR = Path(__file__).parent
DATA_DIR = DUNNHUMBY_DIR / "data"
CACHE_DIR = DUNNHUMBY_DIR / "cache"
OUTPUT_DIR = DUNNHUMBY_DIR / "output"

# Dataset files (from Kaggle ZIP)
TRANSACTION_FILE = DATA_DIR / "transaction_data.csv"
PRODUCT_FILE = DATA_DIR / "product.csv"
DEMOGRAPHICS_FILE = DATA_DIR / "hh_demographic.csv"

# =============================================================================
# PRODUCT CATEGORIES
# =============================================================================

# Top 10 commodity categories to analyze (high-frequency staple goods)
# These are selected for frequent trade-offs between substitutes
# Names must match exactly with COMMODITY_DESC in product.csv
TOP_COMMODITIES = [
    "SOFT DRINKS",
    "FLUID MILK PRODUCTS",
    "BAKED BREAD/BUNS/ROLLS",
    "CHEESE",
    "BAG SNACKS",
    "SOUP",
    "YOGURT",
    "BEEF",
    "FROZEN PIZZA",
    "LUNCHMEAT",
]

# Commodity name mapping for cleaner output
COMMODITY_SHORT_NAMES = {
    "SOFT DRINKS": "Soda",
    "FLUID MILK PRODUCTS": "Milk",
    "BAKED BREAD/BUNS/ROLLS": "Bread",
    "CHEESE": "Cheese",
    "BAG SNACKS": "Chips",
    "SOUP": "Soup",
    "YOGURT": "Yogurt",
    "BEEF": "Beef",
    "FROZEN PIZZA": "Pizza",
    "LUNCHMEAT": "Lunch",
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Household filtering
MIN_SHOPPING_WEEKS = 10  # Minimum weeks for household to be included

# Time parameters (dataset spans 2 years = 104 weeks)
NUM_WEEKS = 104
NUM_PRODUCTS = len(TOP_COMMODITIES)  # 10

# Price filtering (remove outliers)
MIN_UNIT_PRICE = 0.01  # $0.01 minimum
MAX_UNIT_PRICE = 50.0  # $50 maximum per unit

# Algorithm parameters
AEI_TOLERANCE = 1e-4  # Looser tolerance for speed (default is 1e-6)
GARP_TOLERANCE = 1e-10  # Standard GARP tolerance

# =============================================================================
# PERFORMANCE TARGETS
# =============================================================================

# Target: process all households in under 5 minutes
MAX_PROCESSING_TIME_SECONDS = 300

# Progress reporting interval
PROGRESS_INTERVAL = 250  # Print progress every N households

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Rationality thresholds
HIGH_RATIONALITY_THRESHOLD = 0.95  # "Highly rational"
LOW_RATIONALITY_THRESHOLD = 0.70  # "Erratic shopper"
PERFECT_RATIONALITY = 1.0

# Income bracket ordering (for Showcase B)
INCOME_ORDER = [
    "Under 15K",
    "15-24K",
    "25-34K",
    "35-49K",
    "50-74K",
    "75-99K",
    "100-124K",
    "125-149K",
    "150-174K",
    "175-199K",
    "200K+",
]
