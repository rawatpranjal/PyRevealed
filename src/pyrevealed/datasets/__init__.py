"""Dataset loaders for PyRevealed.

Provides sklearn-style dataset loading functions that return BehaviorPanel
objects ready for analysis.

Example:
    >>> from pyrevealed.datasets import load_dunnhumby
    >>> panel = load_dunnhumby()
    >>> print(panel.summary())

Note:
    Data files are NOT bundled with PyRevealed (too large for PyPI).
    You must download them separately. Each loader provides instructions
    when the data is not found.
"""

from pyrevealed.datasets._demo import load_demo
from pyrevealed.datasets._dunnhumby import load_dunnhumby
from pyrevealed.datasets._open_ecommerce import load_open_ecommerce
from pyrevealed.datasets._uci_retail import load_uci_retail
from pyrevealed.datasets._retailrocket import load_retailrocket
from pyrevealed.datasets._instacart import load_instacart
from pyrevealed.datasets._yoochoose import load_yoochoose
from pyrevealed.datasets._olist import load_olist
from pyrevealed.datasets._m5 import load_m5
from pyrevealed.datasets._rees46 import load_rees46


def list_datasets() -> list[dict[str, str]]:
    """List available datasets with descriptions.

    Returns:
        List of dicts with name, description, source, and size info.
    """
    return [
        {
            "name": "demo",
            "description": "Synthetic budget data: 40% rational, 40% noisy, 20% irrational consumers",
            "source": "Generated (seeded, deterministic)",
            "goods": "5 categories (default)",
            "observations": "15 per user (default)",
        },
        {
            "name": "retailrocket",
            "description": "E-commerce click-stream from 1.4M visitors, reconstructed into menu choices",
            "source": "Kaggle (retailrocket/ecommerce-dataset)",
            "goods": "Menu-based (items viewed per session)",
            "observations": "~5-50 sessions per user",
        },
        {
            "name": "dunnhumby",
            "description": "Grocery transactions from 2,500 households over 2 years",
            "source": "Kaggle (dunnhumby - The Complete Journey)",
            "goods": "10 commodity categories",
            "observations": "~50 weeks per household",
        },
        {
            "name": "open_ecommerce",
            "description": "Amazon purchases from ~4,700 consumers over 2018-2022",
            "source": "Open E-Commerce 1.0 (Crowdsourced)",
            "goods": "50 product categories",
            "observations": "~20 months per user",
        },
        {
            "name": "uci_retail",
            "description": "UK online retail transactions from ~1,800 customers",
            "source": "UCI Machine Learning Repository",
            "goods": "50 product categories",
            "observations": "~8 months per customer",
        },
        {
            "name": "instacart",
            "description": "Grocery orders from 200K+ users across 21 departments",
            "source": "Kaggle (instacart/market-basket-analysis)",
            "goods": "21 departments (uniform prices)",
            "observations": "~15-100 orders per user",
        },
        {
            "name": "yoochoose",
            "description": "E-commerce click sessions from RecSys 2015 challenge",
            "source": "RecSys 2015 Challenge",
            "goods": "Menu-based (items clicked per session)",
            "observations": "~5-50 sessions per user",
        },
        {
            "name": "m5",
            "description": "Walmart item sales across 10 stores, 3 states, 1941 days",
            "source": "Kaggle (m5-forecasting-accuracy)",
            "goods": "7 departments",
            "observations": "~277 weeks per store",
        },
        {
            "name": "olist",
            "description": "Brazilian e-commerce orders from 96K customers across marketplaces",
            "source": "Kaggle (olistbr/brazilian-ecommerce)",
            "goods": "20 product categories",
            "observations": "~4-12 months per repeat customer",
        },
        {
            "name": "rees46",
            "description": "Multi-category eCommerce behavior (view/cart/purchase) from Oct-Nov 2019",
            "source": "Kaggle (mkechinov/ecommerce-behavior-data-from-multi-category-store)",
            "goods": "Menu-based (items viewed per session)",
            "observations": "~5-50 sessions per user",
        },
    ]


__all__ = [
    "load_demo",
    "load_dunnhumby",
    "load_open_ecommerce",
    "load_uci_retail",
    "load_retailrocket",
    "load_instacart",
    "load_yoochoose",
    "load_olist",
    "load_m5",
    "load_rees46",
    "list_datasets",
]
