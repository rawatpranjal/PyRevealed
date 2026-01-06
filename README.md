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

## How It Works

PyRevealed implements **revealed preference theory** from economics:

1. **Consistency Check (GARP)**: If a consumer chose bundle A when B was affordable, and chose B when C was affordable, they should not choose C when A was affordable. Violations indicate inconsistency.

2. **Afriat Efficiency Index**: Measures what fraction of behavior is consistent with utility maximization. Score of 1.0 = perfectly consistent; lower scores indicate departures from rationality.

3. **Money Pump Index**: Measures how much money could be extracted from a consumer by exploiting their preference cycles.

The math is based on Afriat (1967) and Varian (1982).

---

## Empirical Study: Dunnhumby Consumer Data

Analysis of the **Dunnhumby "The Complete Journey"** dataset—2 years of grocery transactions from ~2,500 households.

### Dataset Overview

| Metric | Value |
|--------|-------|
| Households analyzed | 2,222 |
| Product categories | 10 (Soda, Milk, Bread, Cheese, Chips, Soup, Yogurt, Beef, Pizza, Lunchmeat) |
| Time period | 104 weeks (2 years) |
| Total transactions | 645,288 |
| Processing time | 92 seconds |

### Methodology

**Integrity Score (AEI)**: If you bought more apples when they were expensive and fewer when cheap, something's off. The integrity score finds the largest fraction of your choices that could come from a single consistent preference. Score of 1.0 = perfectly consistent; 0.5 = random noise.

**WARP (Weak Axiom)**: Did you ever directly contradict yourself? Example: Bought bundle A when B was cheaper, then bought B when A was cheaper.

**GARP (Generalized Axiom)**: Did you contradict yourself through a chain? Example: Preferred A over B, B over C, but then C over A. GARP catches more violations because it follows transitivity chains.

**Money Pump Index (MPI)**: How much money could a clever salesman extract by exploiting your inconsistencies? If you say A > B > C > A, someone can sell you A for B, B for C, C for A—you end up where you started but poorer. The MPI is the fraction of spending that could be "pumped" this way.

**Separability**: Do you have separate mental budgets for different things? If Dairy and Protein are separable, buying more milk doesn't change how you buy beef. We test whether behavior in group A can be explained independently of group B.

### Key Findings

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

### Key Insights

- 4.5% of households are perfectly consistent (GARP)
- Mean integrity score: 0.839
- Mean exploitability (MPI): 0.225
- Only Protein vs Staples shows separate mental budgets (62%)

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
│   └── data/            # Kaggle dataset (download required)
├── docs/images/         # README visualizations
└── notebooks/           # Tutorials
```

## Theory

Based on *Revealed Preference Theory* by Chambers & Echenique (2016):

- **Chapter 2**: Abstract choice consistency (WARP, SARP)
- **Chapter 3**: Afriat's Theorem and rational demand
- **Chapter 5**: Efficiency indices and money pump
- **Chapter 11**: Ideal point models in feature space

## License

MIT
