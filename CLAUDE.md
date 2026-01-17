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

1. **Bump version** in all 3 files (PyPI rejects duplicate versions!):
   ```bash
   # Edit these files with new version X.Y.Z:
   # - pyproject.toml (line 7)
   # - src/pyrevealed/__init__.py (line ~400)
   # - docs/conf.py (line 14)
   ```

2. **Build and upload to PyPI**:
   ```bash
   rm -rf dist/ build/
   python3 -m build
   python3 -m twine upload dist/*
   ```

3. **Rebuild docs** (clean build to avoid caching issues):
   ```bash
   rm -rf docs/_build
   python3 -m sphinx docs docs/_build/html
   ```

4. **Push to GitHub** (triggers ReadTheDocs rebuild):
   ```bash
   git add .
   git commit -m "Release vX.Y.Z"
   git push
   ```

5. **Verify all surfaces**:
   - PyPI: https://pypi.org/project/pyrevealed/
   - ReadTheDocs: https://pyrevealed.readthedocs.io
   - GitHub: https://github.com/rawatpranjal/PyRevealed

### Common Issues

- **PyPI "file already exists"**: You forgot to bump version. PyPI never allows re-uploading the same version.
- **ReadTheDocs not updating**: Push to GitHub triggers rebuild. Wait 1-2 min. If stuck, check build logs at readthedocs.org.
- **Local docs not updating**: Delete `docs/_build/` and rebuild. Sphinx caches aggressively.

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

## Market Opportunity

### Python's Revealed Preference Void

PyRevealed fills a significant gap in Python's scientific ecosystem:

| Language | Package | Status |
|----------|---------|--------|
| **R** | `revealedPrefs` | Active, comprehensive |
| **Stata** | `checkax`, `aei` | Active, enterprise |
| **MATLAB** | Varian toolbox | Active, academic |
| **Python** | PyRevealed | **Only option** |

Before PyRevealed, Python practitioners had to:
- Port R/Stata code manually
- Implement algorithms from scratch
- Use fragmented one-off scripts

### Implementation Coverage

Based on survey of 65+ revealed preference methods from the literature:

| Category | Coverage | Key Methods |
|----------|----------|-------------|
| Consistency Scores | 83% | CCEI, MPI, Swaps, Houtman-Maks |
| Graph Methods | 100% | Floyd-Warshall, cycle detection, centrality |
| Consideration Sets | 80% | WARP-LA, RAM, attention overload |
| Stochastic Choice | 71% | RUM, regularity, IIA |
| Power Analysis | 67% | Bronars power, fast power |
| Welfare Analysis | 60% | CV/EV, cost recovery |
| Preference Bounds | 40% | Afriat bounds (E-bounds/i-bounds missing) |
| Context Effects | 25% | Regularity (decoy/compromise missing) |
| Pairwise/Ranking | 20% | Condorcet (Bradley-Terry missing) |
| Temporal Methods | 0% | Not implemented |

**Overall: ~60% of surveyed methods implemented**

See `docs/implementation_status.md` for detailed gap analysis.

### Key Differentiators

1. **ML-Native Design**: sklearn-compatible API, feature extraction for pipelines
2. **Dual API**: Tech-friendly names (`compute_integrity_score`) + economics terms (`compute_aei`)
3. **Production-Ready**: Type hints, dataclass results, comprehensive tests
4. **Unified Framework**: Budget, menu, stochastic, and production analysis in one package
