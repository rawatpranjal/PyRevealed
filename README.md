# PyRevealed

**Production-ready revealed preference analysis.** Test if choices are internally consistent, quantify behavioral consistency, and analyze decision patterns.

> Based on: Chambers, C. P., & Echenique, F. (2016). *Revealed Preference Theory*. Cambridge University Press.

[![PyPI](https://img.shields.io/pypi/v/pyrevealed)](https://pypi.org/project/pyrevealed/)
[![Documentation](https://readthedocs.org/projects/pyrevealed/badge/?version=latest)](https://pyrevealed.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install pyrevealed
```

For visualization support:
```bash
pip install pyrevealed[viz]
```

## Quick Start

```python
from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
import numpy as np

# Two purchase observations: prices and quantities
log = BehaviorLog(
    cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
    action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
)

# Check if choices are consistent with utility maximization
is_consistent = validate_consistency(log)  # True

# Get integrity score (0 = inconsistent, 1 = perfectly consistent)
result = compute_integrity_score(log)
print(f"Integrity: {result.efficiency_index:.2f}")  # 1.00
```

## Core Functions

| Function | Returns | Score Meaning |
|----------|---------|---------------|
| `validate_consistency(log)` | `bool` | True = rational |
| `compute_integrity_score(log)` | `AEIResult` (0-1) | 1 = perfect |
| `compute_confusion_metric(log)` | `MPIResult` (0-1) | 0 = no cycles |
| `fit_latent_values(log)` | `UtilityRecoveryResult` | Utility values |
| `compute_minimal_outlier_fraction(log)` | `HoutmanMaksResult` (0-1) | 0 = all consistent |

**Quick interpretation**: Integrity >= 0.95 is excellent, >= 0.90 is good, < 0.70 indicates problems.

## Menu-Based Choice

For discrete choices without prices (surveys, recommendations, voting):

```python
from pyrevealed import MenuChoiceLog, validate_menu_sarp, fit_menu_preferences

log = MenuChoiceLog(
    menus=[frozenset({0, 1, 2}), frozenset({1, 2}), frozenset({0, 2})],
    choices=[0, 1, 0],
    item_labels=["Pizza", "Burger", "Salad"]
)

result = validate_menu_sarp(log)
print(f"Consistent: {result.is_consistent}")

prefs = fit_menu_preferences(log)
print(f"Preference order: {prefs.preference_order}")
```

## Documentation

**[Full Documentation](https://pyrevealed.readthedocs.io/)**

### Tutorials

1. **[Budget-Based Analysis](https://pyrevealed.readthedocs.io/en/latest/tutorial.html)** - GARP, CCEI, MPI, Bronars power
2. **[Menu-Based Choice](https://pyrevealed.readthedocs.io/en/latest/tutorial_menu_choice.html)** - WARP, SARP, attention models
3. **[Welfare Analysis](https://pyrevealed.readthedocs.io/en/latest/tutorial_welfare.html)** - CV, EV, deadweight loss
4. **[Demand Analysis](https://pyrevealed.readthedocs.io/en/latest/tutorial_demand_analysis.html)** - Slutsky matrix, integrability
5. **[Stochastic & Production](https://pyrevealed.readthedocs.io/en/latest/tutorial_advanced.html)** - Logit, IIA, firm behavior
6. **[E-Commerce at Scale](https://pyrevealed.readthedocs.io/en/latest/tutorial_ecommerce.html)** - 1.85M Amazon transactions

## Features

- **Consistency Testing**: GARP, WARP, SARP axiom verification
- **Behavioral Metrics**: Afriat Efficiency Index, Money Pump Index
- **Utility Recovery**: Reconstruct utility functions from choices
- **ML Integration**: sklearn-compatible `PreferenceEncoder`
- **Multiple Data Types**: Budgets, menus, stochastic choice, production
- **Production Ready**: Fast parallel processing, validated against R's revealedPrefs

## License

MIT
