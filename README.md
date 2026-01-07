# PyRevealed

A Python implementation of revealed preference theory.

> Based on: Chambers, C. P., & Echenique, F. (2016). *Revealed Preference Theory*. Cambridge University Press.

## What is this?

PyRevealed tests whether observed choices are consistent with utility maximization—and if so, recovers the underlying preferences.

**For economists**: Implements GARP/WARP consistency tests, Afriat's theorem, the Afriat efficiency index, money pump index, and separability tests.

**For data scientists**: Given a history of user choices and the options available at each choice, this library computes:
- **Consistency scores**: How internally consistent is this user's behavior? (0 = random, 1 = perfectly consistent)
- **Preference recovery**: If consistent, what utility function explains their choices?
- **Exploitability metrics**: How much could be extracted from a user via arbitrage on their inconsistencies?
- **Separability tests**: Are choices over group A independent of choices over group B?

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

## Concepts & Tests

PyRevealed implements tests from **revealed preference theory**. Each test analyzes a `BehaviorLog` containing `(prices, quantities)` observations.

### Consistency Tests

#### WARP — Weak Axiom of Revealed Preference

**What it tests**: Direct preference contradictions (no transitivity).

**Definition**: If bundle *x* was chosen when *y* was affordable, then *y* should never be chosen when *x* is affordable.

**Formally**: If *p_t · x_t ≥ p_t · x_s*, then we should not have *p_s · x_s > p_s · x_t*.

**Interpretation**: `True` = no direct contradictions; `False` = at least one direct reversal.

```python
is_consistent, violations = check_warp(log)  # Legacy
result = validate_consistency_weak(log)       # Tech-friendly
```

#### GARP — Generalized Axiom of Revealed Preference

**What it tests**: Transitive preference contradictions (follows chains).

**Definition**: If *x* is directly or indirectly revealed preferred to *y*, then *y* should not be strictly revealed preferred to *x*.

**Formally**: Build revealed preference relation *R* where *x_t R x_s* iff *p_t · x_t ≥ p_t · x_s*. Take transitive closure *R\**. GARP holds iff no cycle exists where strict preference closes the loop.

**Interpretation**: `True` = behavior consistent with utility maximization; `False` = preference cycle exists.

```python
result = check_garp(log)           # Legacy
result = validate_consistency(log)  # Tech-friendly
```

---

### Efficiency Indices

#### AEI — Afriat Efficiency Index (Integrity Score)

**What it measures**: Fraction of behavior explainable by utility maximization.

**Definition**: Find the largest *e ∈ [0, 1]* such that the "wasted" fraction *(1 - e)* of each budget makes all choices GARP-consistent.

**Formally**: Binary search for max *e* where *e · p_t · x_t ≥ p_t · x_s* admits no violations.

**Interpretation**:
- `1.0` = perfectly consistent
- `0.9+` = minor noise
- `0.7–0.9` = moderate inconsistency
- `<0.7` = significant departures (possible bot/shared account)

```python
result = compute_aei(log)              # Legacy
result = compute_integrity_score(log)  # Tech-friendly
score = result.efficiency_index        # Float in [0, 1]
```

#### MPI — Money Pump Index (Confusion Metric)

**What it measures**: Exploitability via preference cycles.

**Definition**: Maximum fraction of spending extractable by cycling through preference contradictions.

**Formally**: Find worst cycle *i₁ → i₂ → ... → iₖ → i₁* and compute money extractable per round.

**Interpretation**:
- `0.0` = unexploitable (fully consistent)
- `0.1–0.3` = minor exploitability
- `>0.3` = significant confusion (bad UX or manipulation vulnerability)

```python
result = compute_mpi(log)                 # Legacy
result = compute_confusion_metric(log)    # Tech-friendly
score = result.mpi_value                  # Float in [0, 1]
```

#### Houtman-Maks Index (Outlier Fraction)

**What it measures**: Minimum observations to remove for GARP consistency.

**Definition**: Smallest set of observations whose removal makes remaining data GARP-consistent.

**Interpretation**:
- `0.0` = already consistent
- `<0.1` = "almost rational" (few outliers)
- `>0.2` = substantial inconsistency

```python
fraction, removed_indices = compute_minimal_outlier_fraction(log)
```

#### VEI — Varian Efficiency Index (Per-Observation Efficiency)

**What it measures**: Efficiency score for each individual observation.

**Definition**: Solve LP to minimize *Σ(1 - eᵢ)* subject to GARP constraints with observation-specific efficiency *eᵢ*.

**Formally**: *eᵢ · (pᵢ · xᵢ) ≥ pᵢ · xⱼ* for all *(i, j)* in transitive closure.

**Interpretation**: Identifies which specific observations are problematic (low *eᵢ*).

```python
result = compute_vei(log)                   # Legacy
result = compute_granular_integrity(log)    # Tech-friendly
result.efficiency_vector                    # Array of per-obs scores
result.problematic_observations             # Indices where eᵢ < threshold
```

---

### Statistical Power

#### Bronars' Power Index (Test Power)

**What it measures**: Statistical significance of passing GARP.

**Definition**: Fraction of random behaviors (uniform on budget hyperplanes) that would violate GARP.

**Formally**: Generate *N* random bundles via Dirichlet distribution on each budget set. Power = violation rate.

**Interpretation**:
- `>0.5` = test is statistically meaningful
- `<0.5` = even random behavior passes (low power, uninformative)

```python
result = compute_bronars_power(log, n_simulations=1000)  # Legacy
result = compute_test_power(log, n_simulations=1000)     # Tech-friendly
result.power_index          # Float in [0, 1]
result.is_significant       # True if power > 0.5
```

---

### Preference Structure

#### HARP — Homothetic Axiom (Proportional Scaling)

**What it tests**: Do preferences scale proportionally with budget?

**Definition**: Relative spending shares should remain constant regardless of income level.

**Formally**: For expenditure ratio *rᵢⱼ = (pᵢ · xᵢ) / (pᵢ · xⱼ)*, product around any cycle must be ≤ 1.

**Interpretation**: `True` = homothetic preferences (Cobb-Douglas, CES); `False` = income effects on composition.

```python
result = check_harp(log)                        # Legacy
result = validate_proportional_scaling(log)     # Tech-friendly
result.is_homothetic                            # Boolean
```

#### Quasilinearity (Income Invariance)

**What it tests**: Is marginal utility of money constant?

**Definition**: Choices depend only on relative prices, not income level.

**Formally**: For any cycle, *Σₖ pₖ · (xₖ₊₁ - xₖ) ≥ 0* (cyclic monotonicity).

**Interpretation**: `True` = no income effects; `False` = behavior varies with budget.

```python
result = check_quasilinearity(log)      # Legacy
result = test_income_invariance(log)    # Tech-friendly
result.is_quasilinear                   # Boolean
result.has_income_effects               # Inverse property
```

#### Separability (Feature Independence)

**What it tests**: Can preferences over group A be analyzed independently of group B?

**Definition**: Choices within group A don't depend on quantities in group B.

**Formally**: Submatrix of choices restricted to group A satisfies GARP independently.

**Interpretation**: `True` = separate "mental budgets"; `False` = cross-group dependencies.

```python
result = check_separability(log, group_a=[0, 1], group_b=[2, 3])  # Legacy
result = test_feature_independence(log, group_a=[0, 1], group_b=[2, 3])  # Tech-friendly
```

---

### Cross-Price Effects

#### Gross Substitutes / Complements

**What it tests**: Relationship between goods when prices change.

**Definition**:
- **Substitutes**: Price of A ↑ → demand for B ↑ (replace A with B)
- **Complements**: Price of A ↑ → demand for B ↓ (bought together)

**Formally**: Compare quantity changes when price of good *g* changes while others stay constant.

```python
result = check_gross_substitutes(log, good_g=0, good_h=1)  # Legacy
result = test_cross_price_effect(log, good_g=0, good_h=1)  # Tech-friendly
result.relationship  # "substitutes", "complements", "independent", "inconclusive"

# Full matrix
matrix = compute_cross_price_matrix(log)
```

---

## Empirical Study: Dunnhumby Consumer Data

Application of PyRevealed to the **Dunnhumby "The Complete Journey"** dataset—2 years of grocery transactions from ~2,500 households.

### Dataset Overview

| Metric | Value |
|--------|-------|
| Households analyzed | 2,222 |
| Product categories | 10 (Soda, Milk, Bread, Cheese, Chips, Soup, Yogurt, Beef, Pizza, Lunchmeat) |
| Time period | 104 weeks (2 years) |
| Total transactions | 645,288 |
| Processing time | 92 seconds |

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **GARP-consistent** | 100 (4.5%) | Only 4.5% of households are perfectly rational |
| **Mean integrity score** | 0.839 | Most behavior is ~84% consistent |
| **Median integrity score** | 0.855 | Half of households score above 0.855 |
| **Low consistency** (< 0.7) | 285 (12.8%) | 12.8% show significant departures from rationality |
| **Perfect rationality** (1.0) | 100 (4.5%) | Utility recovery possible for these |

### Rationality Distribution

Most households cluster around 0.85 integrity:

![Rationality Distribution](docs/images/showcase_a_rationality_histogram.png)

### Exploitability (Money Pump Index)

How much could be extracted from users via arbitrage on their inconsistencies?

![MPI Distribution](docs/images/analysis_mpi_distribution.png)

Mean MPI: 0.225. Strong negative correlation with integrity (r=-0.89).

### Mental Accounting

Do households treat product groups as separate budgets?

![Mental Accounting](docs/images/showcase_e_mental_accounting.png)

Only Protein (Beef, Lunchmeat) vs Staples (Bread, Soup) shows separate budgeting (62%). All other category pairs: pooled budgets (<35%).

### Auto-Discovered Product Groups

Does the data confirm our manual category groupings? Using `discover_independent_groups()`:

| Manual Group | Products | Auto-Discovery Agreement |
|--------------|----------|-------------------------|
| **Dairy** | Milk, Cheese, Yogurt | 61% ✓ |
| **Snacks** | Soda, Chips, Pizza | 60% ✓ |
| **Staples** | Bread, Soup | 71% ✓ |
| **Protein** | Beef, Lunchmeat | 42% ✗ |

Three of four manual groupings confirmed. Protein products don't cluster as strongly as expected.

### Robustness (Houtman-Maks Index)

How many observations need to be removed to make behavior consistent?

| Metric | Value |
|--------|-------|
| Mean outlier fraction | 18.6% |
| Perfect rationality (HM=0) | 3.8% |
| Easy fixes (HM<0.1) | 17.0% |
| AEI-HM correlation | -0.499 |

17% of households are "almost rational"—removing <10% of their observations makes them fully consistent.

### Preference Structure Tests

| Test | Result | Interpretation |
|------|--------|----------------|
| **Bronars Power** | 0.845 (87.5% significant) | GARP test has high statistical power |
| **Homotheticity (HARP)** | 3.2% pass | Few scale preferences with budget |
| **Income Invariance** | 0% quasilinear | All show income effects |
| **Per-Obs Efficiency (VEI)** | 0.534 mean | Granular efficiency per observation |

### Cross-Price Relationships

| Product Pair | Score | Relationship |
|--------------|-------|--------------|
| Milk & Bread | -0.31 | Complements |
| Soda & Pizza | -0.29 | Complements |
| Milk & Cheese | -0.29 | Complements |
| Cheese & Lunchmeat | -0.23 | Complements |
| Soda & Chips | -0.22 | Complements |

Most product pairs show complementary relationships (bought together), consistent with grocery shopping patterns.

![Cross-Price Matrix](docs/images/showcase_o_cross_price.png)

### Predictive Validation

Can first-half behavior predict second-half outcomes? Split-sample study with LightGBM:

| Target | Persistence RMSE | LightGBM RMSE | R² | Improvement |
|--------|------------------|---------------|-----|-------------|
| **Integrity (AEI)** | 0.114 | 0.081 | 0.084 | **28.4%** |
| **Total Spending** | 195.78 | 195.03 | 0.785 | 0.4% |
| **Category Shares** | 0.06-0.09 | 0.06-0.09 | 0.33-0.55 | 1-5% |

**Ablation Study: Incremental Value of PyRevealed Features**

| Feature Set | Integrity R² | Spending R² |
|-------------|--------------|-------------|
| Basic only (14 features) | 0.071 | 0.784 |
| Basic + PyRevealed (24 features) | **0.084** | 0.785 |
| **PyRevealed R² lift** | **+0.014** | +0.001 |

PyRevealed features (BehavioralAuditor + PreferenceEncoder) contribute **12.5% of total feature importance** for integrity prediction. First-half `integrity_score` ranks **#5** among all predictors of second-half consistency.

### Key Insights

| Category | Finding |
|----------|---------|
| **Consistency** | 4.5% GARP-consistent, mean AEI = 0.839 |
| **Exploitability** | Mean MPI = 0.225 |
| **Statistical Power** | Bronars = 0.845 (87.5% significant) |
| **Preference Structure** | 3.2% homothetic, 0% quasilinear |
| **Separability** | Only Protein vs Staples shows separate budgets (62%) |
| **Robustness** | 17% "almost rational" (HM < 0.1) |
| **Cross-Price** | Mostly complements (Milk+Bread, Soda+Pizza) |
| **Prediction** | PyRevealed features: +0.014 R² lift (12.5% importance) |

### Running the Dunnhumby Tests

```bash
# 1. Download the Kaggle dataset (requires kaggle CLI)
cd dunnhumby && ./download_data.sh

# 2. Run the full integration test suite
python3 dunnhumby/run_all.py

# 3. Run extended analysis (income, spending, time trends)
python3 dunnhumby/extended_analysis.py

# 4. Run comprehensive analysis (MPI, WARP, separability)
python3 dunnhumby/comprehensive_analysis.py

# 5. Run advanced analysis (complementarity, mental accounting, stress test, structural breaks)
python3 dunnhumby/advanced_analysis.py

# 6. Run encoder analysis (auto-discovery, Houtman-Maks)
python3 dunnhumby/encoder_analysis.py

# 7. Run predictive validation (split-sample LightGBM)
python3 dunnhumby/predictive_analysis.py

# 8. Run new algorithms analysis (Bronars, HARP, VEI, quasilinearity, cross-price)
python3 dunnhumby/new_algorithms_analysis.py

# Optional: Quick test mode (100 households sample)
python3 dunnhumby/run_all.py --quick
```

---

## Project Structure

```
pyrevealed/
├── src/pyrevealed/
│   ├── auditor.py       # BehavioralAuditor class
│   ├── encoder.py       # PreferenceEncoder class
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

## Theory

Based on *Revealed Preference Theory* by Chambers & Echenique (2016):

- **Chapter 2**: Abstract choice consistency (WARP, SARP)
- **Chapter 3**: Afriat's Theorem and rational demand
- **Chapter 5**: Efficiency indices and money pump
- **Chapter 11**: Ideal point models in feature space

## License

MIT
