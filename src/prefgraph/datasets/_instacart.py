"""Instacart Market Basket dataset loader.

Loads the Instacart "Market Basket Analysis" dataset and aggregates
orders at the aisle level (134 aisles). Since individual product prices
are not available, heuristic per-aisle prices are assigned based on
aisle names (keyword matching to price tiers). This gives meaningful
price variation for revealed preference analysis.

Data must be downloaded separately from Kaggle:
  kaggle datasets download -d instacart/market-basket-analysis
  unzip market-basket-analysis.zip -d ~/.prefgraph/data/instacart/

Source: https://www.kaggle.com/c/instacart-market-basket-analysis
License: Competition-specific (research use)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prefgraph.core.panel import BehaviorPanel
from prefgraph.core.session import BehaviorLog


# ---------------------------------------------------------------------------
# Heuristic price tiers: keyword -> $/unit
# ---------------------------------------------------------------------------
# Ordered so that more specific keywords match first. The lookup function
# iterates top-to-bottom and returns the price for the first keyword hit.
_PRICE_TIERS: list[tuple[list[str], float]] = [
    # Alcohol (most expensive tier)
    (["spirit"], 14.00),
    (["champagne", "specialty wine"], 12.00),
    (["red wine", "white wine"], 10.00),
    (["beer", "cooler"], 8.00),

    # Protein (meat, seafood, poultry)
    (["seafood counter"], 8.00),
    (["packaged seafood", "canned meat seafood"], 5.50),
    (["meat counter", "packaged meat"], 6.00),
    (["poultry counter", "packaged poultry"], 5.50),
    (["hot dog", "bacon", "sausage"], 5.00),
    (["lunch meat"], 4.50),
    (["frozen meat"], 6.00),

    # Health / personal care / baby
    (["vitamin", "supplement"], 8.00),
    (["baby food", "formula"], 5.00),
    (["baby accessor"], 6.00),
    (["baby bath"], 5.50),
    (["diaper", "wipe"], 7.00),
    (["first aid"], 6.00),
    (["cold flu", "allergy", "muscles joints", "pain relief"], 6.50),
    (["oral hygiene"], 4.50),
    (["hair care"], 6.00),
    (["skin care", "facial care"], 6.50),
    (["body lotion", "soap"], 5.50),
    (["deodorant"], 4.50),
    (["shave"], 5.50),
    (["feminine care"], 5.00),
    (["eye ear care"], 5.50),
    (["beauty"], 6.00),
    (["digestion"], 6.00),

    # Cheese
    (["specialty cheese"], 6.00),
    (["packaged cheese", "other cream", "cheese"], 5.00),

    # Dairy / eggs
    (["ice cream"], 4.50),
    (["cream"], 3.50),
    (["butter"], 3.50),
    (["yogurt"], 3.00),
    (["milk"], 3.00),
    (["soy lactosefree"], 3.50),
    (["egg"], 3.00),
    (["pudding"], 3.00),

    # Frozen meals / pizza / appetizers
    (["frozen pizza"], 5.00),
    (["frozen meal"], 4.50),
    (["frozen appetizer", "frozen side"], 4.00),
    (["frozen breakfast"], 3.50),
    (["frozen bread", "frozen dough"], 3.50),
    (["frozen vegan", "frozen vegetarian"], 4.50),
    (["frozen juice"], 3.00),
    (["frozen produce"], 3.00),
    (["frozen dessert"], 4.00),

    # Bakery / bread
    (["bakery dessert"], 4.00),
    (["bread"], 3.00),
    (["bun", "roll", "tortilla", "flat bread"], 3.00),
    (["breakfast bar", "pastri"], 3.50),
    (["breakfast bakery"], 3.00),

    # Bulk bins (before produce/snacks so "bulk" catches these first)
    (["bulk"], 3.00),

    # Fresh produce
    (["fresh fruit"], 2.50),
    (["fresh vegetable"], 2.00),
    (["fresh herb"], 2.50),
    (["fresh pasta"], 3.00),
    (["fresh dip", "tapenade"], 3.50),
    (["packaged produce", "packaged vegetable", "packaged fruit"], 3.00),

    # Pantry condiments (before beverages so "honey/syrup" beats "nectar")
    (["honey", "syrup"], 3.50),

    # Beverages -- specific multi-word matches before generic "energy"/"sport"
    (["energy granola"], 3.00),
    (["energy sport"], 3.00),
    (["protein", "meal replacement"], 5.00),
    (["juice", "nectar"], 3.00),
    (["water", "seltzer", "sparkling"], 2.00),
    (["soft drink"], 2.50),
    (["coffee"], 4.00),
    (["tea"], 3.00),
    (["cocoa", "drink mix"], 3.00),

    # Snacks
    (["chip", "pretzel"], 3.50),
    (["popcorn", "jerky"], 3.50),
    (["cookie", " cake"], 3.50),
    (["candy", "chocolate"], 3.00),
    (["cracker"], 3.00),
    (["mint", "gum"], 2.00),
    (["trail mix", "snack mix"], 3.50),
    (["fruit vegetable snack"], 3.00),
    (["nut", "seed", "dried fruit"], 3.50),
    (["granola"], 3.50),

    # Pantry staples
    (["canned meal", "bean"], 2.00),
    (["canned jarred vegetable"], 2.00),
    (["canned fruit", "applesauce"], 2.00),
    (["prepared soup", "prepared salad"], 3.50),
    (["soup", "broth", "bouillon"], 2.50),
    (["prepared meal"], 4.50),
    (["pasta sauce"], 2.50),
    (["dry pasta"], 1.50),
    (["grain", "rice", "dried good"], 2.50),
    (["baking ingredient", "baking supplie", "baking decor"], 2.50),
    (["dough", "gelatin", "bake mix"], 2.50),
    (["spice", "season"], 3.00),
    (["condiment"], 2.50),
    (["salad dressing", "topping"], 3.00),
    (["oil", "vinegar"], 3.50),
    (["spread"], 3.00),
    (["preserved dip"], 3.00),
    (["pickle", "olive"], 3.00),
    (["marinade", "meat preparation"], 3.00),
    (["hot cereal", "pancake mix"], 3.00),
    (["cereal"], 3.50),
    (["instant food"], 2.50),
    (["tofu", "meat alternative"], 3.50),

    # International foods
    (["latino"], 2.50),
    (["asian"], 2.50),
    (["indian"], 2.50),
    (["kosher"], 3.00),

    # Household
    (["cleaning product"], 4.00),
    (["dish detergent"], 3.50),
    (["laundry"], 5.00),
    (["trash bag", "liner"], 4.00),
    (["paper good"], 4.50),
    (["air freshener", "candle"], 3.50),
    (["food storage"], 3.50),
    (["plate", "bowl", "cup", "flatware"], 3.00),
    (["kitchen supplie"], 3.00),
    (["more household"], 3.50),

    # Pets
    (["dog food", "dog care"], 5.00),
    (["cat food", "cat care"], 5.00),
    (["pet"], 5.00),

    # Catch-all for "refrigerated", "other", "missing", etc.
    (["refrigerated"], 3.00),
    (["other"], 3.00),
    (["missing"], 3.00),
]

_DEFAULT_PRICE = 3.00


def _aisle_price(aisle_name: str) -> float:
    """Return heuristic $/unit price for an aisle based on keyword matching."""
    name = aisle_name.lower()
    for keywords, price in _PRICE_TIERS:
        if any(kw in name for kw in keywords):
            return price
    return _DEFAULT_PRICE


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find Instacart data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "instacart")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "instacart",
        Path(__file__).resolve().parents[3] / "datasets" / "instacart" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "orders.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Instacart data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle:\n"
        "  kaggle datasets download -d instacart/market-basket-analysis\n"
        "  unzip market-basket-analysis.zip -d ~/.prefgraph/data/instacart/\n\n"
        "Required files: orders.csv, order_products__prior.csv, products.csv, aisles.csv"
    )


def load_instacart(
    data_dir: str | Path | None = None,
    max_users: int | None = None,
    min_orders: int = 10,
) -> BehaviorPanel:
    """Load Instacart dataset as a BehaviorPanel.

    Aggregates products at the aisle level (134 aisles). Uses heuristic
    per-aisle prices based on keyword matching of aisle names (e.g.
    fresh produce ~$2, meat/seafood ~$6, alcohol ~$10).

    Args:
        data_dir: Path to directory containing Instacart CSV files.
        max_users: Maximum number of users (None = all).
        min_orders: Minimum orders per user (default 10).

    Returns:
        BehaviorPanel with one BehaviorLog per user.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'prefgraph[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)

    print(f"  Loading Instacart data from {data_path}...")

    # Load orders -- use only "prior" (main historical data)
    orders = pd.read_csv(data_path / "orders.csv")
    prior_orders = orders[orders["eval_set"] == "prior"].copy()
    prior_orders = prior_orders.sort_values(["user_id", "order_number"])

    # Load order-product details
    order_products = pd.read_csv(data_path / "order_products__prior.csv")

    # Load product -> aisle mapping
    products = pd.read_csv(data_path / "products.csv")
    aisles = pd.read_csv(data_path / "aisles.csv")
    products = products.merge(aisles, on="aisle_id")

    # Build heuristic price lookup: aisle_id -> $/unit
    aisle_price_map = {
        row["aisle_id"]: _aisle_price(row["aisle"])
        for _, row in aisles.iterrows()
    }

    # Merge to get aisle_id per order-product
    order_products = order_products.merge(
        products[["product_id", "aisle_id"]],
        on="product_id",
    )

    # Count items per aisle per order
    aisle_counts = (
        order_products
        .groupby(["order_id", "aisle_id"])
        .size()
        .reset_index(name="quantity")
    )

    # Merge with order info to get user_id and order_number
    aisle_counts = aisle_counts.merge(
        prior_orders[["order_id", "user_id", "order_number"]],
        on="order_id",
    )

    # Filter users with enough orders
    user_order_counts = prior_orders.groupby("user_id")["order_id"].nunique()
    qualifying_users = user_order_counts[user_order_counts >= min_orders].index

    if max_users is not None:
        qualifying_users = qualifying_users[:max_users]

    aisle_counts = aisle_counts[aisle_counts["user_id"].isin(qualifying_users)]

    # Build column index from observed aisles
    aisle_ids = sorted(aisle_counts["aisle_id"].unique())
    aisle_idx = {a: i for i, a in enumerate(aisle_ids)}
    n_cols = len(aisle_ids)

    # Constant price vector (same for every observation)
    price_vector = np.array([aisle_price_map[a] for a in aisle_ids])

    # Build per-user BehaviorLogs
    logs: dict[str, BehaviorLog] = {}

    for user_id, user_data in aisle_counts.groupby("user_id"):
        order_nums = sorted(user_data["order_number"].unique())
        T = len(order_nums)
        if T < min_orders:
            continue

        qty_matrix = np.zeros((T, n_cols))
        for _, row in user_data.iterrows():
            t_idx = order_nums.index(row["order_number"])
            a_idx = aisle_idx[row["aisle_id"]]
            qty_matrix[t_idx, a_idx] += row["quantity"]

        # Replicate price vector across all observations
        price_matrix = np.tile(price_vector, (T, 1))

        uid = f"user_{user_id}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    # Build aisle name list for metadata
    aisle_names = []
    aisle_name_map = dict(zip(aisles["aisle_id"], aisles["aisle"]))
    for a in aisle_ids:
        aisle_names.append(aisle_name_map.get(a, f"aisle_{a}"))

    price_range = (price_vector.min(), price_vector.max())
    print(
        f"  Built {len(logs)} BehaviorLog objects "
        f"({n_cols} aisles, prices ${price_range[0]:.2f}-${price_range[1]:.2f}/unit)"
    )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "instacart",
            "goods": aisle_ids,
            "aisle_names": aisle_names,
            "n_aisles": n_cols,
            "price_type": "heuristic_per_aisle",
            "price_range": price_range,
        },
    )
