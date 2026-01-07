# Datasets for PyRevealed Validation

This directory contains data loaders for publicly available datasets that can be used to validate PyRevealed's revealed preference algorithms.

## Available Datasets

| Dataset | Source | Users | Periods | Goods | License |
|---------|--------|-------|---------|-------|---------|
| **UCI Online Retail** | [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) | ~1,800 | 13 months | 50 products | CC BY 4.0 |
| **Open E-Commerce 1.0** | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YGLYDY) | ~4,700 | 66 months | 50 categories | CC0 |

## Quick Start

### Install dependencies

```bash
pip install ucimlrepo openpyxl requests pyarrow
```

### UCI Online Retail

```bash
# Download (22 MB)
python datasets/uci_retail/download.py

# Run quick validation
python datasets/uci_retail/run_validation.py --quick

# Run full validation
python datasets/uci_retail/run_validation.py
```

### Open E-Commerce 1.0

```bash
# Download (299 MB)
python datasets/open_ecommerce/download.py

# Run quick validation
python datasets/open_ecommerce/run_validation.py --quick

# Run full validation
python datasets/open_ecommerce/run_validation.py
```

## Usage in Python

```python
from datasets.uci_retail.session_builder import load_sessions

# Load customer sessions
customers = load_sessions()

# Run GARP validation on a single customer
from pyrevealed import validate_consistency, compute_integrity_score

customer = customers[list(customers.keys())[0]]
result = validate_consistency(customer.behavior_log)
print(f"GARP consistent: {result.is_consistent}")

if not result.is_consistent:
    aei = compute_integrity_score(customer.behavior_log)
    print(f"Efficiency index: {aei.efficiency_index:.4f}")
```

## Dataset Details

### UCI Online Retail

- **Source**: UK-based online retail store transactions (2010-2011)
- **Format**: Invoice-level transaction data with customer IDs
- **Goods**: Top 50 products by transaction volume
- **Filtering**: Removes cancelled orders, missing customer IDs, invalid prices

### Open E-Commerce 1.0

- **Source**: Crowdsourced Amazon purchase histories from 5,027 US consumers (2018-2022)
- **Format**: Individual order-level data with survey response IDs
- **Goods**: Top 50 Amazon product categories
- **Reference**: Goldfarb, Tucker, & Wang (2024). *Scientific Data*, 11(527).

## Directory Structure

```
datasets/
├── README.md
├── cache/                        # Parquet cache files (gitignored)
├── uci_retail/
│   ├── config.py                 # Dataset-specific constants
│   ├── download.py               # Download from UCI
│   ├── data_loader.py            # Load and preprocess
│   ├── session_builder.py        # Build BehaviorLog objects
│   └── run_validation.py         # Quick validation script
└── open_ecommerce/
    ├── config.py                 # Dataset-specific constants
    ├── download.py               # Download from Harvard Dataverse
    ├── data_loader.py            # Load and preprocess
    ├── session_builder.py        # Build BehaviorLog objects
    └── run_validation.py         # Quick validation script
```

## Results Summary

Sample validation results:

| Dataset | Customers | GARP Consistent | Mean AEI |
|---------|-----------|-----------------|----------|
| UCI Online Retail | 1,819 | ~100% | 1.00 |
| Open E-Commerce 1.0 | 4,744 | ~13% | 0.88 |

The high consistency rate for UCI Online Retail reflects the B2B wholesale nature of the data. The lower rate for Open E-Commerce reflects the heterogeneity of consumer preferences across diverse product categories.
