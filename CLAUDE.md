# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Rules

- **Always commit and push after completing changes.** Do not wait for the user to ask — commit and push to main automatically when work is done.

## Documentation Rules

- **RTD nav tab names must be single words.** Current order: Quickstart, Budgets, Menus, Algorithms, Performance, Applications, Methods, API, References. Methods/API/References are intentionally pushed right so they overflow into "More" on narrow viewports. Applications wraps Production + Intertemporal.
- **README is user-forward.** Paste-and-run example first (using `load_demo`), then Scores, "Which API?", Choice Categories, Performance. Architecture details live in RTD only.
- **CHANGELOG.md** lives at repo root. Update it when making user-visible changes.

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
python3 case_studies/dunnhumby/run_all.py --quick   # 100 households sample
python3 case_studies/dunnhumby/run_all.py           # Full 2,222 households

# Rust rebuild (maturin is the build backend)
pip install -e .                        # Recompiles Rust automatically
maturin develop --release               # Direct maturin (faster iteration)
```

The Rust binding crate is at `rust/crates/rpt-python/` (not repo root).
`pip install .` compiles Rust via maturin; `HAS_RUST` fallback in `_rust_backend.py`
still works if the Rust toolchain is unavailable.

## Applications
- to showcase the powerful rust engine batch processing
- to showcase REAL WORLD APPLICATIONS (zero simulated, real data and concrete usecase)
- real outputs, run the script and put in real results please.

### Batch-First Rule (Engine-Default)
Multi-user scoring MUST use `Engine.analyze_arrays()` (budget) or `Engine.analyze_menus()` (discrete).
No Python for-loops calling per-user algorithm functions (`validate_consistency`, `validate_menu_sarp`,
`compute_integrity_score`, `compute_menu_efficiency`, etc.) over collections of users.

The Engine handles parallelism via Rust/Rayon. Use `log.to_engine_tuple()` for conversion.

Acceptable single-user patterns:
- Rolling-window temporal analysis (per-user window slicing)
- Single-user pipeline walkthroughs for educational purposes
- Small-N contexts (<10 subjects, e.g., 5 LLM prompts)

## Architecture

PyRevealed tests whether observed choices are consistent with rational optimization — without estimating parameters. We test existence ("does a utility function exist?"), not fit models ("which utility function?"). All graph algorithms and LP solving run in Rust via Rayon; Python handles I/O.

### 4 Choice Categories × 4 Method Types

Primary axis: **data type**. Secondary axis: what the method does.

| | Test (bool) | Score (0→1) | Recover (vector) | Structure (bool) |
|---|---|---|---|---|
| **Budget** (prices × quantities) | GARP, WARP, GAPP, Slutsky | CCEI, MPI, HM, VEI, Swaps | Utility, Demand, CV/EV | HARP, Quasilinear, Separability |
| **Discrete** (menus × choices) | SARP, WARP, WARP-LA, RUM LP, IIA | HM (menu), Regularity | Ordinal utility | Congruence |
| **Production** (inputs × outputs) | Prod GARP | Prod CCEI | Tech efficiency | Cost min, Returns to scale |
| **Intertemporal** (dated amounts) | Exp discounting | — | Discount factor δ | Quasi-hyperbolic |

"Discrete" unifies menu choice, stochastic choice, and risk — all "pick from a set" with different item types.

### Full Method Reference with Citations

**Budget Choice:**
- GARP: SCC + Floyd-Warshall — Varian (1982) *Econometrica*
- CCEI/AEI: Binary search over T² ratios — Afriat (1967) *IER*
- MPI: Karp's max-mean-weight cycle — Echenique, Lee & Shum (2011) *JPE*
- Houtman-Maks: Greedy FVS / ILP — Houtman & Maks (1985)
- VEI: Per-observation LP — Varian (1990) *J Econometrics*
- HARP: Log-space Floyd-Warshall — Varian (1983) *RES*
- Quasilinear: Bellman-Ford negative cycles — Chambers & Echenique (2016) Ch 9
- GAPP: FW on price preferences — Deb, Kitamura, Quah & Stoye (2023) *RES*
- Utility Recovery: Afriat LP via HiGHS — Afriat (1967)
- CV/EV Welfare: Afriat LP + expenditure — Vartia (1983) *Econometrica*
- Swaps Index: Greedy FAS — Apesteguia & Ballester (2015) *JPE*
- Min Cost Index: Cycle cost — Dean & Martin (2016) *REStat*

**Discrete Choice:**
- SARP: FW on item graph — Richter (1966) *Econometrica*
- WARP-LA: Consideration sets — Masatlioglu, Nakajima & Ozbay (2012) *AER*
- RUM LP: LP on K! orderings — Block & Marschak (1960); Kitamura & Stoye (2018)
- Regularity: Subset dominance — Debreu (1960)
- Congruence: SARP + maximality — Richter (1966)

**Production Choice:**
- Production GARP: FW on profit graph — Varian (1984) *Econometrica*

**Intertemporal Choice:**
- Exponential Discounting: Bound propagation — Echenique, Imai & Saito (2020) *AEJ:Micro*
- Quasi-Hyperbolic: Grid search + LP — Laibson (1997) *QJE*

### Rust Engine (rpt-core)

```
User data (any of 4 types)
    ↓
PreferenceGraph.parse_*()     ← builds expenditure/preference matrices
    ↓
Algorithms (graph + LP)       ← SCC, Floyd-Warshall, HiGHS LP/ILP
    ↓
Rayon par_iter (batch)        ← one thread per user, scratchpad reuse
    ↓
Engine results
```

### Backend Parity

Python fallback (`Engine._analyze_chunk_python`) supports GARP + CCEI + MPI only.
HARP, HM, utility, and VEI require the Rust backend.

Known algorithm differences:
- **MPI**: Python uses GARP-cycle enumeration; Rust uses Karp's max-mean-weight cycle. Tolerance: 0.05
- **CCEI**: Both use discrete binary search. Tolerance: 0.01
- **HM**: Both use greedy FVS with SCC recomputation. Should match exactly.

Cross-backend parity tests: `pytest tests/test_backend_parity.py`

### Core Data Flow

```
BehaviorLog (prices × quantities)             DiscreteChoiceLog (menus × choices/frequencies)
    ↓                                              ↓
┌──────────────────────────────────────┐  ┌────────────────────────────────────┐
│ Budget Choice (Rust)                 │  │ Discrete Choice (Rust)             │
│  Test:  GARP, WARP, GAPP            │  │  Test:  SARP, WARP, WARP-LA,      │
│  Score: CCEI, MPI, HM, VEI          │  │         RUM LP, IIA               │
│  Recover: Utility, Demand, CV/EV    │  │  Score: HM (menu)                 │
│  Structure: HARP, Separability      │  │  Recover: Ordinal utility          │
└──────────────────────────────────────┘  │  Structure: Congruence, Regularity│
                                          └────────────────────────────────────┘
ProductionLog (inputs × outputs)          Intertemporal data (amounts × dates)
    ↓                                          ↓
┌──────────────────────────────────────┐  ┌────────────────────────────────────┐
│ Production Choice (Rust)             │  │ Intertemporal Choice (Rust)        │
│  Test:  Prod GARP                    │  │  Test:  Exp discounting            │
│  Score: Prod CCEI                    │  │  Recover: Discount factor δ        │
│  Structure: Cost min, RTS            │  │  Structure: Quasi-hyperbolic       │
└──────────────────────────────────────┘  └────────────────────────────────────┘
└───────────────────────────────────────┘
```

### Key Modules (algorithms/)

| Module | Category | Computation |
|---|---|---|
| garp.py | Budget | SCC + Floyd-Warshall — Varian (1982) |
| aei.py | Budget | Discrete binary search — Afriat (1967) |
| mpi.py | Budget | Karp's max-mean cycle — Echenique+ (2011) |
| harp.py | Budget | Log-space FW — Varian (1983) |
| utility.py | Budget | Afriat LP — Afriat (1967) |
| vei.py | Budget | Per-obs efficiency LP — Varian (1990) |
| abstract_choice.py | Discrete | FW on item graph — Richter (1966) |
| attention.py | Discrete | WARP-LA — Masatlioglu+ (2012) |
| stochastic.py | Discrete | RUM LP — Kitamura & Stoye (2018) |
| production.py | Production | Profit graph FW — Varian (1984) |
| quasilinear.py | Budget | Bellman-Ford — C&E (2016) Ch 9 |
| intertemporal.py | Intertemporal | Bound propagation — Echenique+ (2020) |

### Result Types (`core/results/`)

57 result dataclasses split into submodules by domain:

| Module | Key Classes |
|--------|-------------|
| `budget_core.py` | GARPResult, AEIResult, MPIResult, UtilityRecoveryResult |
| `budget_extended.py` | WARPResult, SARPResult, HoutmanMaksResult, HARPResult, VEIResult, ... |
| `abstract_choice.py` | AbstractWARPResult, CongruenceResult, OrdinalUtilityResult, ... |
| `advanced.py` | IntegrabilityResult, WelfareResult, ProductionGARPResult, ... |
| `diagnostics.py` | RegularityResult, SwapsIndexResult, BradleyTerryResult, ... |
| `power.py` | SeltenMeasureResult, RelativeAreaResult, OptimalEfficiencyResult, ... |
| `risk.py` | RiskProfileResult, ExpectedUtilityResult |
| `spatial.py` | IdealPointResult, SeparabilityResult |
| `attention.py` | WARPLAResult, RandomAttentionResult, RUMConsistencyResult |

`from pyrevealed.core.result import XxxResult` still works (backward-compat shim).
31 tech-friendly aliases (e.g. `ConsistencyResult = GARPResult`) live in each submodule.

### Contrib (deprecated — MLE/estimation)

Located in `src/pyrevealed/contrib/`. Logit/Luce MLE, CRRA estimation, Bradley-Terry,
Slutsky regression, Spatial ideal point. Import shims with DeprecationWarning.

### Rust Engine (rpt-core)

```
rust/crates/rpt-core/src/
├── garp.rs, ccei.rs, mpi.rs, harp.rs     ← Budget (Test + Score)
├── utility.rs, vei.rs, welfare.rs         ← Budget (Recover)
├── quasilinear.rs, additive.rs            ← Budget (Structure)
├── separability.rs, gapp.rs, variants.rs  ← Budget (Structure)
├── menu.rs, attention.rs, stochastic.rs   ← Discrete
├── production.rs                           ← Production
├── intertemporal.rs                        ← Intertemporal
├── graph.rs, closure.rs, scc.rs            ← Infrastructure
├── lp.rs, houtman_maks.rs                 ← Shared (LP, FVS)
└── 57 tests
```

### Production Data Flow

```
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

### Quick Deploy Commands

When the user says **"release"**, **"deploy"**, or **"push to all surfaces"**, run these commands:

```bash
# 1. Stage and commit all changes
git add -A
git commit -m "Description of changes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 2. Push to GitHub (triggers RTD rebuild automatically)
git push

# 3. Verify surfaces after 1-2 minutes:
#    - ReadTheDocs: https://pyrevealed.readthedocs.io
#    - GitHub: https://github.com/rawatpranjal/PyRevealed
```

**Note**: This does NOT push to PyPI. For a full PyPI release, follow the Release Checklist above (requires version bump).

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
