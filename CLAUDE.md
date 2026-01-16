# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install
pip install .                    # Basic install
pip install ".[dev]"             # With dev tools (pytest, mypy, ruff)
pip install ".[viz]"             # With visualization (matplotlib)

# Test
pytest                           # Run all tests with coverage
pytest tests/test_garp.py        # Single test file
pytest -k "test_consistent"      # Run tests matching pattern

# Type check & lint
mypy src/
ruff check src/
ruff format src/

# Real-world validation (requires Kaggle dataset)
python3 dunnhumby/run_all.py --quick   # 100 households sample
python3 dunnhumby/run_all.py           # Full 2,222 households
```

## Architecture

PyRevealed implements revealed preference theory to analyze behavioral consistency.

### Core Data Flow

```
BehaviorLog (prices + quantities)     MenuChoiceLog (menus + choices)
    ↓                                      ↓
┌───────────────────────────────────────┐  ┌─────────────────────────────────┐
│ Core Algorithms (algorithms/)         │  │ Abstract Choice (algorithms/)   │
│  • garp.py → consistency check        │  │  • abstract_choice.py           │
│  • aei.py → integrity score (0-1)     │  │    → WARP/SARP/Congruence       │
│  • mpi.py → exploitability metric     │  │    → Houtman-Maks efficiency    │
│  • utility.py → preference recovery   │  │    → Ordinal utility recovery   │
│  • separability.py → group independence│  │  • attention.py                 │
└───────────────────────────────────────┘  │    → Limited attention models   │
    ↓                                      └─────────────────────────────────┘
Result dataclasses (core/result.py)             ↓
                                         StochasticChoiceLog (frequencies)
                                               ↓
                                         ┌─────────────────────────────────┐
                                         │ Stochastic (algorithms/)        │
                                         │  • stochastic.py                │
                                         │    → Random utility models      │
                                         │    → IIA/regularity tests       │
                                         └─────────────────────────────────┘

ProductionLog (inputs + outputs)
    ↓
┌───────────────────────────────────────┐
│ Production (algorithms/)              │
│  • production.py                      │
│    → Profit maximization test         │
│    → Cost minimization check          │
│    → Returns to scale estimation      │
└───────────────────────────────────────┘

Advanced Analysis (algorithms/)
┌───────────────────────────────────────┐
│  • integrability.py → Slutsky tests   │
│  • welfare.py → CV/EV computation     │
│  • additive.py → additive separability│
│  • gross_substitutes.py → Slutsky     │
│    decomposition, Hicksian demand     │
│  • spatial.py → general metric prefs  │
└───────────────────────────────────────┘
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `auditor.py` | High-level `BehavioralAuditor` class (linter-style API) |
| `encoder.py` | `PreferenceEncoder` and `MenuPreferenceEncoder` for ML feature extraction |
| `core/session.py` | `BehaviorLog`, `MenuChoiceLog`, `StochasticChoiceLog`, `ProductionLog` containers |
| `algorithms/garp.py` | GARP consistency via Floyd-Warshall transitive closure |
| `algorithms/aei.py` | Afriat Efficiency Index via binary search |
| `algorithms/mpi.py` | Money Pump Index via cycle detection |
| `algorithms/utility.py` | Utility recovery via scipy.linprog |
| `algorithms/abstract_choice.py` | Menu-based WARP/SARP/Congruence, Houtman-Maks, ordinal utility |
| `algorithms/integrability.py` | Slutsky symmetry/NSD tests (Ch 6) |
| `algorithms/welfare.py` | Compensating/equivalent variation (Ch 7) |
| `algorithms/additive.py` | Additive separability tests (Ch 9) |
| `algorithms/gross_substitutes.py` | Slutsky decomposition, Hicksian demand (Ch 10) |
| `algorithms/spatial.py` | General metric preference recovery (Ch 11) |
| `algorithms/stochastic.py` | Random utility models, IIA tests (Ch 13) |
| `algorithms/attention.py` | Limited attention, consideration sets (Ch 14) |
| `algorithms/production.py` | Profit/cost tests for firm behavior (Ch 15) |

### API Pattern

Primary API uses tech-friendly names:
```python
from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score, compute_confusion_metric
from pyrevealed import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency, fit_menu_preferences
```

Old economics names still work as aliases:
```python
from pyrevealed import ConsumerSession, check_garp, compute_aei, compute_mpi
from pyrevealed import check_abstract_sarp, check_congruence, recover_ordinal_utility
```

### Test Fixtures

Test data in `tests/conftest.py`:
- `simple_consistent_session` - 3 observations, GARP-consistent
- `simple_violation_session` - 2 observations, WARP violation
- `three_cycle_violation_session` - 3-cycle GARP violation
- `large_random_session` - 100 obs × 10 goods for performance

## Version Alignment

When releasing a new version, these files must be kept in sync:

| File | Field | Example |
|------|-------|---------|
| `pyproject.toml` | `version = "X.Y.Z"` | Line 7 |
| `src/pyrevealed/__init__.py` | `__version__ = "X.Y.Z"` | Line 262 |
| `docs/conf.py` | `release = "X.Y.Z"` | Line 14 |

### Verification Commands

```bash
# Check all versions match
grep -E "^version|__version__|release" pyproject.toml src/pyrevealed/__init__.py docs/conf.py

# Verify module version
python3 -c "import pyrevealed; print(pyrevealed.__version__)"

# Check PyPI version
pip index versions pyrevealed

# Check URLs are correct
grep -n "github" pyproject.toml docs/conf.py
```

### Release Checklist

1. Update version in all 3 files above
2. Ensure GitHub URLs point to `https://github.com/rawatpranjal/PyRevealed`
3. Ensure `LICENSE` file exists
4. Ensure Python version in `docs/tutorial*.rst` matches `pyproject.toml` (`requires-python`)
5. Commit, push, and verify:
   - PyPI: https://pypi.org/project/pyrevealed/
   - ReadTheDocs: https://pyrevealed.readthedocs.io
   - GitHub: https://github.com/rawatpranjal/PyRevealed

## Theory Reference

Based on Chambers & Echenique (2016) *Revealed Preference Theory*:

**Budget-Based (Chapters 3-5):**
- GARP: Generalized Axiom of Revealed Preference (transitivity check)
- AEI: Afriat Efficiency Index (fraction of behavior consistent with utility maximization)
- MPI: Money Pump Index (exploitability via preference cycles)

**Menu-Based / Abstract Choice (Chapters 1-2):**
- WARP: Weak Axiom (no direct preference reversals)
- SARP: Strong Axiom (no preference cycles of any length)
- Congruence: Full rationalizability (SARP + maximality)
- Houtman-Maks: Fraction of observations that are consistent

**Advanced Analysis (Chapters 6-15):**
- Integrability (Ch 6): Slutsky symmetry and negative semi-definiteness
- Welfare (Ch 7): Compensating and equivalent variation
- Additive Separability (Ch 9): No cross-price effects
- Compensated Demand (Ch 10): Slutsky decomposition, Hicksian demand
- General Metrics (Ch 11): Ideal point with non-Euclidean distances
- Stochastic Choice (Ch 13): Random utility models, IIA, regularity
- Limited Attention (Ch 14): Consideration sets, attention filters
- Production (Ch 15): Profit maximization, cost minimization tests
