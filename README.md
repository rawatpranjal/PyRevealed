# PyRevealed

Non-parametric behavioral consistency metrics for fraud detection and user segmentation.

**PyRevealed** implements revealed preference theory algorithms to analyze consumer behavior consistency. Unlike machine learning models that predict *what* users will buy, PyRevealed tells you if their behavior is *consistent* with rational decision-making.

## Key Features

- **GARP Detection**: Check if consumer data satisfies the Generalized Axiom of Revealed Preference
- **Afriat Efficiency Index (AEI)**: Score behavioral consistency from 0 (irrational) to 1 (perfectly rational)
- **Money Pump Index (MPI)**: Measure exploitable inconsistencies in choices
- **Utility Recovery**: Reconstruct the utility function rationalizing observed behavior

## Installation

```bash
pip install pyrevealed
```

For visualization support:
```bash
pip install pyrevealed[viz]
```

For Jupyter notebook examples:
```bash
pip install pyrevealed[notebooks]
```

## Quick Start

```python
import numpy as np
from pyrevealed import ConsumerSession, check_garp, compute_aei

# Create session with price and quantity data
# prices: T observations x N goods
# quantities: T observations x N goods
prices = np.array([
    [1.0, 2.0],  # Observation 0: price of good A=1, good B=2
    [2.0, 1.0],  # Observation 1: price of good A=2, good B=1
])
quantities = np.array([
    [3.0, 1.0],  # Observation 0: bought 3 of A, 1 of B
    [1.0, 3.0],  # Observation 1: bought 1 of A, 3 of B
])

session = ConsumerSession(prices=prices, quantities=quantities)

# Check GARP consistency
result = check_garp(session)
print(f"Is consistent: {result.is_consistent}")

# Get efficiency score
aei_result = compute_aei(session)
print(f"Efficiency Index: {aei_result.efficiency_index:.2f}")
```

## Use Cases

### Bot Detection
Bots often generate clicks/views randomly, failing transitive consistency tests:
```python
score = compute_aei(user_session)
if score.efficiency_index < 0.85:
    flag_as_potential_bot(user_id)
```

### Account Sharing Detection
Shared accounts (e.g., Netflix profiles used by multiple people) exhibit inconsistent preferences:
```python
monthly_scores = [compute_aei(month).efficiency_index for month in user_history]
if np.mean(monthly_scores) < 0.90:
    prompt_profile_split(user_id)
```

### UI/UX A/B Testing
Measure if a new UI confuses users into making irrational choices:
```python
control_mpi = np.mean([compute_mpi(s).mpi_value for s in control_group])
variant_mpi = np.mean([compute_mpi(s).mpi_value for s in variant_group])
if variant_mpi > control_mpi * 1.5:
    rollback_ui_change()
```

## API Reference

### Data Structures

- `ConsumerSession(prices, quantities)` - Core data container
- `GARPResult` - Result of GARP consistency check
- `AEIResult` - Afriat Efficiency Index result
- `MPIResult` - Money Pump Index result

### Functions

- `check_garp(session)` - Test GARP consistency
- `compute_aei(session)` - Calculate efficiency index
- `compute_mpi(session)` - Calculate money pump index
- `recover_utility(session)` - Recover utility function via LP

## Theory

Based on *Revealed Preference Theory* by Christopher P. Chambers and Federico Echenique. The package implements:

- **Chapter 3**: Rational Demand and Afriat's Theorem
- **Chapter 5**: Afriat Efficiency Index and Money Pump Index

## License

MIT
