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


def list_datasets() -> list[dict[str, str]]:
    """List available datasets with descriptions.

    Returns:
        List of dicts with name, description, source, and size info.
    """
    return [
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
    ]


__all__ = [
    "load_demo",
    "load_dunnhumby",
    "load_open_ecommerce",
    "load_uci_retail",
    "load_retailrocket",
    "list_datasets",
]
