# Case Study: Dunnhumby Consumer Data

Real-world validation of PyRevealed on 2 years of grocery transactions from 2,222 households.

**For the full tutorial with step-by-step code, see the [Tutorial Documentation](https://pyrevealed.readthedocs.io/en/latest/tutorial.html).**

## Key Findings

| Category | Finding |
|----------|---------|
| **Consistency** | 4.5% perfectly consistent, mean integrity = 0.839 |
| **Exploitability** | Mean confusion = 0.225 |
| **Mental Accounting** | Only Protein vs Staples shows separate budgets (62%) |
| **Cross-Price** | Mostly complements (Milk+Bread, Soda+Pizza) |
| **Lancaster Model** | 5.4% "rescued" in characteristics-space |
| **Smooth Preferences** | 1.6% differentiable |
| **Price Preferences** | 0% GAPP pass |

## Quick Start

```bash
# Download the dataset
cd dunnhumby && ./download_data.sh

# Run the full analysis
python3 dunnhumby/run_all.py
```

## Documentation

- [Full Tutorial](https://pyrevealed.readthedocs.io/en/latest/tutorial.html) - Step-by-step walkthrough with code
- [API Reference](https://pyrevealed.readthedocs.io/en/latest/api.html) - Complete function documentation
- [Case Study Summary](https://pyrevealed.readthedocs.io/en/latest/case_study.html) - Results overview
