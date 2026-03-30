# PrefGraph

Rationality scores for every user. Rust engine, Python interface.

[![PyPI](https://img.shields.io/pypi/v/prefgraph)](https://pypi.org/project/prefgraph/)
[![Documentation](https://readthedocs.org/projects/prefgraph/badge/?version=latest)](https://prefgraph.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

```bash
pip install prefgraph
```

## Quick Example

Score how consistently each user's choices align with rational utility maximization:

```python
from prefgraph.datasets import load_demo
from prefgraph.engine import Engine

# 100 synthetic shoppers (prices x quantities), no download needed
users = load_demo()

# Engine scores every user in parallel via Rust/Rayon
engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
results = engine.analyze_arrays(users)

for r in results[:3]:
    print(r)
```

```
EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (42us)
EngineResult: [-] 3 violations  ccei=0.8472  mpi=0.0231  hm=12/15  (38us)
EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (35us)
```

Every score is a feature you can plug into downstream models. In our benchmarks across 11 e-commerce datasets, revealed preference features rank among the top ten by model importance but add near-zero marginal lift over well-constructed baselines. See the [case studies](https://prefgraph.readthedocs.io/en/latest/benchmarks_ecommerce.html) for details.

## Your Data

Have a DataFrame? One line:

```python
import prefgraph as rp

# Transaction logs (one row per user x time x item)
results = rp.analyze(df, user_col="user_id", item_col="product",
                     cost_col="price", action_col="quantity", time_col="week")

# Wide format (one row per observation, goods as columns)
results = rp.analyze(df, user_col="user_id",
                     cost_cols=["price_A", "price_B"],
                     action_cols=["qty_A", "qty_B"])

# Menu/click data (which items were shown, which was picked)
results = rp.analyze(df, user_col="user_id",
                     menu_col="shown_items", choice_col="clicked")
```

Returns a DataFrame with one row per user. Customize with `metrics=["garp", "ccei", "mpi", "hm"]`.

## Scores

| Score | Field | What it measures | Range |
|-------|-------|-----------------|-------|
| Consistency | `is_garp` | Are choices rationalizable? (GARP) | bool |
| Efficiency | `ccei` | How close to perfectly rational? (Afriat) | 0-1 |
| Exploitability | `mpi` | Value left on the table per choice (Karp cycle) | 0-1 |
| Homotheticity | `is_harp` | Do preferences scale with budget? | bool |
| Rationalizable fraction | `hm_consistent/hm_total` | Fraction of rationalizable choices (Houtman-Maks) | 0-1 |
| Utility recovery | `utility_success` | Can latent utility be reconstructed? (Afriat LP) | bool |
| Per-obs efficiency | `vei_mean` | Average efficiency across observations (Varian) | 0-1 |

## Which API?

| | Engine | Function API |
|---|---|---|
| Use case | Score thousands of users | Deep-dive one user |
| Speed | 2,000-49,000 users/sec (Rust) | Single-user |
| Returns | `EngineResult` (flat scores) | `GARPResult`, `AEIResult`, etc. (matrices, cycles, graphs) |
| Metrics | 7 (garp, ccei, mpi, harp, hm, utility, vei) + network | 30+ algorithms |
| Input | `list[(prices, quantities)]` | `BehaviorLog` |

**Engine** for batch scoring. **Function API** when you need violation details, observation/item graphs, or advanced tests:

```python
from prefgraph import BehaviorLog, validate_consistency, compute_integrity_score

# 3 shopping trips, 2 goods
session = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

# GARP: does a consistent utility function exist?
garp = validate_consistency(session)       # GARPResult with violation cycles, matrices

# CCEI: how much must budgets shrink to remove contradictions?
ccei = compute_integrity_score(session)    # AEIResult with binary search details
```

## Budget and Menu Choices

PrefGraph supports two primary choice domains:

| Category | Input format | Example domain | Key tests |
|----------|-------------|----------------|-----------|
| **Budget** | `(prices T x K, quantities T x K)` | E-commerce, grocery, food delivery | GARP, CCEI, MPI, HM, HARP, VEI |
| **Discrete** | `(menus, choices)` or `(menus, frequencies)` | Surveys, A/B tests, recommendations, LLM eval | SARP, WARP, HM, RUM LP, IIA |

## Performance

The Rust engine (`rpt-core`) handles graph algorithms and LP solving via Rayon thread pool. Python handles I/O and the user-facing API.

| Configuration | Throughput | 10K users | 100K users |
|---------------|-----------|-----------|------------|
| GARP only | ~49,000/s | 0.1s | 2.0s |
| GARP + CCEI | ~2,400/s | 4.2s | 39.5s |
| Comprehensive (GARP, CCEI, MPI, HARP) | ~2,000/s | 6.8s | 67.1s |
| Menu (SARP + WARP + HM) | ~1,900/s | 0.3s | 5.2s |

Memory stays flat via streaming chunks.

## Documentation

Full docs at [prefgraph.readthedocs.io](https://prefgraph.readthedocs.io/):

- [Loading Data](https://prefgraph.readthedocs.io/en/latest/quickstart.html) - Parquet, DataFrame, synthetic generators
- [Case Studies](https://prefgraph.readthedocs.io/en/latest/benchmarks.html) - LLM consistency, e-commerce benchmarks
- [Algorithms](https://prefgraph.readthedocs.io/en/latest/algorithms.html) - Implementation details
- [API Reference](https://prefgraph.readthedocs.io/en/latest/api.html) - Full method reference

## License

MIT
