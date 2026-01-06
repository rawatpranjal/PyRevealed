# PyRevealed

Non-parametric behavioral consistency metrics for fraud detection and user segmentation.

**PyRevealed** implements revealed preference theory algorithms to analyze consumer behavior consistency. Unlike machine learning models that predict *what* users will buy, PyRevealed tells you if their behavior is *consistent* with rational decision-making.

## Key Features

### Budgetary Data Analysis (Price/Quantity)
- **GARP Detection**: Check if consumer data satisfies the Generalized Axiom of Revealed Preference
- **Afriat Efficiency Index (AEI)**: Score behavioral consistency from 0 (irrational) to 1 (perfectly rational)
- **Money Pump Index (MPI)**: Measure exploitable inconsistencies in choices
- **Houtman-Maks Index**: Minimum observations to remove for consistency
- **Utility Recovery**: Reconstruct the utility function rationalizing observed behavior

### Menu Choice Analysis (Discrete Choice)
- **WARP Check**: Weak Axiom of Revealed Preference for choice functions
- **SARP Check**: Strong Axiom with cycle detection
- **Default Bias Analysis**: Measure tendency to stick with default options

## Installation

```bash
pip install -e .
```

For visualization support:
```bash
pip install -e ".[viz]"
```

For Jupyter notebook examples:
```bash
pip install -e ".[notebooks]"
```

## Quick Start

### Budgetary Data (Prices + Quantities)

```python
import numpy as np
from pyrevealed import ConsumerSession, check_garp, compute_aei, compute_mpi

# Create session with price and quantity data
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

# Get exploitability measure
mpi_result = compute_mpi(session)
print(f"Money Pump Index: {mpi_result.mpi_value:.2f}")
```

### Run Analysis on Prest Datasets

```bash
python scripts/analyze_prest_data.py
```

This loads 11 datasets from the [prest](https://github.com/prestsoftware/prest) project and runs:
- GARP, AEI, MPI, Houtman-Maks on budgetary data
- WARP, SARP, default bias on menu choice data
- Generates visualizations (AEI distribution, preference heatmaps, budget sets)

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

### Default Effect Analysis
Measure how defaults influence choice rationality:
```python
# Compare WARP violations with vs without defaults
# Our analysis shows: 0% WARP-consistent with defaults vs 6% without
```

## Project Structure

```
pyrevealed/
├── src/pyrevealed/
│   ├── algorithms/      # GARP, AEI, MPI, utility recovery
│   ├── core/            # ConsumerSession, RiskSession, SpatialSession
│   ├── graph/           # Transitive closure, violation detection
│   └── viz/             # Plotting utilities
├── scripts/
│   └── analyze_prest_data.py   # Full analysis script
├── notebooks/           # Jupyter tutorials
├── tests/               # Unit tests
└── sim/                 # Simulation studies
```

## API Reference

### Data Structures

| Class | Description |
|-------|-------------|
| `ConsumerSession(prices, quantities)` | Budgetary choice data |
| `RiskSession(safe, risky_outcomes, probs, choices)` | Risk preference data |
| `SpatialSession(features, choice_sets, choices)` | Ideal point analysis |

### Functions

| Function | Description |
|----------|-------------|
| `check_garp(session)` | Test GARP consistency |
| `compute_aei(session)` | Calculate Afriat Efficiency Index (0-1) |
| `compute_mpi(session)` | Calculate Money Pump Index |
| `compute_houtman_maks_index(session)` | Min observations to remove |
| `recover_utility(session)` | Recover utility function via LP |

### Visualization

```python
from pyrevealed.viz import plot_aei_distribution, plot_budget_sets
from pyrevealed.viz.plots import plot_preference_heatmap

# Plot AEI distribution across users
plot_aei_distribution(scores)

# Plot budget sets and choices
plot_budget_sets(session)

# Plot revealed preference matrix
plot_preference_heatmap(session, matrix_type='transitive')
```

## Example Results

Analysis of [prest example datasets](https://github.com/prestsoftware/prest):

### Budgetary Data
| Subject | Observations | GARP | AEI | MPI |
|---------|--------------|------|-----|-----|
| subject1 | 10 | ✓ | 1.000 | 0.00 |
| subject2 | 11 | ✗ | 0.999 | 0.05 |
| subject3 | 10 | ✗ | 0.947 | 0.33 |
| subject4 | 20 | ✗ | 0.778 | 0.23 |

### Menu Choice Data
| Dataset | Subjects | WARP Consistent | SARP Consistent |
|---------|----------|-----------------|-----------------|
| integrity | 2 | 100% | 100% |
| general_defaults | 1000 | 0% | 0% |
| general_no_defaults | 1000 | 6% | 5% |

**Key finding:** Default options dramatically increase WARP violations.

## Theory

Based on *Revealed Preference Theory* by Christopher P. Chambers and Federico Echenique:

- **Chapter 2**: Classical Abstract Choice Theory (WARP, SARP)
- **Chapter 3**: Rational Demand and Afriat's Theorem (GARP)
- **Chapter 5**: Afriat Efficiency Index and Money Pump Index

## License

MIT
