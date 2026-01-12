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
BehaviorLog (prices + quantities)
    ↓
┌───────────────────────────────────────┐
│ Core Algorithms (algorithms/)         │
│  • garp.py → consistency check        │
│  • aei.py → integrity score (0-1)     │
│  • mpi.py → exploitability metric     │
│  • utility.py → preference recovery   │
│  • separability.py → group independence│
└───────────────────────────────────────┘
    ↓
Result dataclasses (core/result.py)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `auditor.py` | High-level `BehavioralAuditor` class (linter-style API) |
| `encoder.py` | `PreferenceEncoder` for ML feature extraction (sklearn-style) |
| `core/session.py` | `BehaviorLog` data container (prices × quantities matrix) |
| `algorithms/garp.py` | GARP consistency via Floyd-Warshall transitive closure |
| `algorithms/aei.py` | Afriat Efficiency Index via binary search |
| `algorithms/mpi.py` | Money Pump Index via cycle detection |
| `algorithms/utility.py` | Utility recovery via scipy.linprog |

### API Pattern

Primary API uses tech-friendly names:
```python
from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score, compute_confusion_metric
```

Old economics names still work as aliases:
```python
from pyrevealed import ConsumerSession, check_garp, compute_aei, compute_mpi
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
- GARP: Generalized Axiom of Revealed Preference (transitivity check)
- AEI: Afriat Efficiency Index (fraction of behavior consistent with utility maximization)
- MPI: Money Pump Index (exploitability via preference cycles)
