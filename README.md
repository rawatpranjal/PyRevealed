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

### Consistency Tests (Boolean)

| Function | What it tests | True means |
|----------|---------------|------------|
| `validate_consistency(log)` | Transitive preference cycles | Utility-maximizing behavior |
| `validate_consistency_weak(log)` | Direct preference contradictions | No direct reversals |
| `validate_sarp(log)` | Indifference cycles | No mutual preferences |
| `validate_smooth_preferences(log)` | Differentiable utility | Can compute elasticities |
| `validate_strict_consistency(log)` | Strict cycles only (lenient) | Approximately rational |
| `validate_price_preferences(log)` | Price preference consistency | Seeks lower prices |

### Efficiency Scores (0–1)

| Function | What it measures | Interpretation |
|----------|------------------|----------------|
| `compute_integrity_score(log)` | Fraction consistent with utility max | 1.0 = perfect, <0.7 = bot risk |
| `compute_confusion_metric(log)` | Exploitability via preference cycles | 0.0 = safe, >0.3 = confused |
| `compute_minimal_outlier_fraction(log)` | Observations to remove for consistency | <0.1 = almost rational |
| `compute_granular_integrity(log)` | Per-observation efficiency | Identifies problem observations |
| `compute_test_power(log)` | Statistical significance of tests | >0.5 = meaningful result |

### Preference Structure (Boolean)

| Function | What it tests | True means |
|----------|---------------|------------|
| `validate_proportional_scaling(log)` | Preferences scale with budget | Homothetic (Cobb-Douglas) |
| `test_income_invariance(log)` | Constant marginal utility of money | No income effects |
| `test_feature_independence(log, groups)` | Group A independent of group B | Separate mental budgets |
| `test_cross_price_effect(log, g, h)` | Substitute/complement relationship | Returns relationship type |

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

## More Examples

See the `examples/` directory for advanced usage:

| File | Description |
|------|-------------|
| `01_behavioral_auditor.py` | Linter-style API with bot/fraud/UX risk assessments |
| `02_preference_encoder.py` | sklearn-style ML integration and counterfactual predictions |
| `03_risk_analysis.py` | Risk profiling under uncertainty (risk-averse vs risk-seeking) |
| `04_spatial_preferences.py` | Ideal point analysis in embedding space |
| `05_advanced_features.py` | Separability, data loading, temporal analysis |
| `06_characteristics_model.py` | Lancaster model for attribute-level preferences |

## Theory

Based on *Revealed Preference Theory* by Chambers & Echenique (2016):

- **Chapter 2**: Abstract choice consistency (WARP, SARP)
- **Chapter 3**: Afriat's Theorem and rational demand
- **Chapter 5**: Efficiency indices and money pump
- **Chapter 11**: Ideal point models in feature space

Also incorporates algorithms from the 2024 survey paper:

> "Revealed preference and revealed preference cycles: A survey" (2024)

- **Differentiable Rationality**: Chiappori & Rochet (1987)
- **Acyclical P**: Dziewulski (2023)
- **GAPP**: Deb et al. (2022)

## License

MIT
