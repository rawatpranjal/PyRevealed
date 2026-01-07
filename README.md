# PyRevealed

A Python implementation of revealed preference theory.

> Based on: Chambers, C. P., & Echenique, F. (2016). *Revealed Preference Theory*. Cambridge University Press.

## What is this?

Given a history of user choices and the options available at each choice, PyRevealed computes:

- **Consistency scores**: How internally consistent is this user's behavior? (0 = random, 1 = perfectly consistent)
- **Preference recovery**: If consistent, what utility function explains their choices?
- **Exploitability metrics**: How much could be extracted from a user via arbitrage on their inconsistencies?
- **Feature independence**: Are choices over group A independent of choices over group B?

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
from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score, compute_confusion_metric
import numpy as np

# Create a behavior log from observed choices
log = BehaviorLog(
    cost_vectors=np.array([      # Prices at each observation (T x N)
        [1.0, 2.0],              # Observation 0: price of good A=1, B=2
        [2.0, 1.0],              # Observation 1: price of good A=2, B=1
    ]),
    action_vectors=np.array([    # Quantities chosen (T x N)
        [3.0, 1.0],              # Observation 0: bought 3 of A, 1 of B
        [1.0, 3.0],              # Observation 1: bought 1 of A, 3 of B
    ])
)

# Test consistency (GARP)
is_consistent = validate_consistency(log)
print(f"Consistent: {is_consistent}")

# Compute integrity score (Afriat Efficiency Index)
integrity = compute_integrity_score(log)
print(f"Integrity Score: {integrity:.3f}")

# Compute confusion metric (Money Pump Index)
confusion = compute_confusion_metric(log)
print(f"Confusion Metric: {confusion:.3f}")
```

---

## Available Tests & Scores

### Yes/No Tests

| Method | Question it answers |
|--------|---------------------|
| `validate_consistency(log)` | Is this user rational? (no self-contradicting choices) |
| `validate_consistency_weak(log)` | Any obvious flip-flops? (picked A over B, then B over A) |
| `validate_smooth_preferences(log)` | Smooth preferences? (needed for price sensitivity analysis) |
| `validate_strict_consistency(log)` | Approximately rational? (ignores minor contradictions) |
| `validate_price_preferences(log)` | Does user prefer situations where their items are cheaper? |

### Scores (0 to 1)

| Method | What it tells you | How to read it |
|--------|-------------------|----------------|
| `compute_integrity_score(log)` | How consistent is this user? | 1.0 = perfect, 0.8 = good, <0.7 = suspicious |
| `compute_confusion_metric(log)` | How exploitable via pricing tricks? | 0 = safe, >0.3 = easily manipulated |
| `compute_minimal_outlier_fraction(log)` | How many bad choices to ignore? | <0.1 = almost perfect, >0.2 = messy data |
| `compute_test_power(log)` | Is "consistent" meaningful here? | >0.5 = yes, <0.5 = test is too easy |

### Preference Structure

| Method | Question it answers |
|--------|---------------------|
| `validate_proportional_scaling(log)` | Do they buy the same mix regardless of budget size? |
| `test_income_invariance(log)` | Does budget size affect what they choose? |
| `test_feature_independence(log, [a], [b])` | Are choices in group A separate from group B? |
| `test_cross_price_effect(log, item1, item2)` | Are these items substitutes or complements? |
| `transform_to_characteristics(log, A)` | Analyze by attributes (nutrition, specs) not products |

---

## Case Study

See **[DUNNHUMBY.md](DUNNHUMBY.md)** for a real-world validation on 2,222 households from the Dunnhumby grocery dataset.

Key findings: 4.5% fully consistent, mean integrity 0.839, test power 0.845.

---

## Project Structure

```
pyrevealed/
├── src/pyrevealed/
│   ├── auditor.py       # BehavioralAuditor class
│   ├── encoder.py       # PreferenceEncoder class
│   ├── lancaster.py     # Lancaster characteristics model
│   ├── algorithms/      # Core algorithms
│   ├── core/            # Data containers
│   ├── graph/           # Graph algorithms
│   └── viz/             # Visualization
├── tests/               # Unit tests
├── dunnhumby/           # Real-world validation suite
│   ├── run_all.py       # Main test runner
│   ├── extended_analysis.py  # Statistical analyses
│   ├── comprehensive_analysis.py  # MPI, WARP, separability
│   ├── advanced_analysis.py  # Complementarity, stress tests
│   ├── encoder_analysis.py  # Auto-discovery, Houtman-Maks
│   ├── predictive_analysis.py  # Split-sample LightGBM
│   ├── lancaster_analysis.py  # Lancaster characteristics model
│   └── data/            # Kaggle dataset (download required)
├── docs/images/         # README visualizations
├── notebooks/           # Tutorials
└── examples/            # Advanced usage examples
```

## License

MIT
