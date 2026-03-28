# PrefGraph

Rationality scores for every user, at scale. Rust engine, Python interface.

[![PyPI](https://img.shields.io/pypi/v/prefgraph)](https://pypi.org/project/prefgraph/)
[![Documentation](https://readthedocs.org/projects/prefgraph/badge/?version=latest)](https://prefgraph.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

```bash
pip install "prefgraph[datasets]"
```

## Quick Example

Score how consistently each user's choices align with rational utility maximization. Paste and run:

```python
from prefgraph.datasets import load_demo
from prefgraph.engine import Engine

users = load_demo()  # 100 synthetic consumers, no download needed
engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
results = engine.analyze_arrays(users)

for r in results[:5]:
    print(r)
```

```
EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (42us)
EngineResult: [-] 3 violations  ccei=0.8472  mpi=0.0231  hm=12/15  (38us)
EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (35us)
...
```

Every score is a feature. Use them for fraud detection, user segmentation, A/B testing, churn prediction, or personalization.

## Your Data

Have a pandas DataFrame? One line:

```python
import prefgraph as rp

# Transaction logs (one row per item per time)
results = rp.analyze(df, user_col="user_id", item_col="product",
                     cost_col="price", action_col="quantity", time_col="week")

# Wide format (one row per observation, items as columns)
results = rp.analyze(df, user_col="user_id",
                     cost_cols=["price_A", "price_B"],
                     action_cols=["qty_A", "qty_B"])

# Menu/click data
results = rp.analyze(df, user_col="user_id",
                     menu_col="shown_items", choice_col="clicked")
```

Returns a pandas DataFrame with one row per user. Customize with `metrics=["garp", "ccei", "mpi", "hm"]`. Handle missing data with `nan_policy="drop"`.

## Scores

| Score | Field | What it measures | Range |
|-------|-------|-----------------|-------|
| Consistency | `is_garp` | Are choices rationalizable? (GARP) | bool |
| Efficiency | `ccei` | How close to perfectly rational? (Afriat) | 0-1 |
| Exploitability | `mpi` | Value left on the table per choice (Karp cycle) | 0-1 |
| Homotheticity | `is_harp` | Do preferences scale with budget? | bool |
| Rationalizable fraction | `hm_consistent/hm_total` | Fraction of rationalizable choices (Houtman–Maks) | 0-1 |
| Utility recovery | `utility_success` | Can latent utility be reconstructed? (Afriat LP) | bool |
| Per-obs efficiency | `vei_mean` | Average efficiency across observations (Varian) | 0-1 |

## Which API?

| | Engine | Function API |
|---|---|---|
| Use case | Score thousands of users | Deep-dive one user |
| Speed | 10,000+ users/sec (Rust) | Single-user |
| Returns | `EngineResult` (flat scores) | `GARPResult`, `AEIResult`, etc. (matrices, cycles, graphs) |
| Metrics | 7 (garp, ccei, mpi, harp, hm, utility, vei) + network | 30+ algorithms |
| Input | `list[(prices, quantities)]` | `BehaviorLog` |

**Engine** for batch scoring. **Function API** when you need violation details, observation/item graphs, or advanced tests:

```python
from prefgraph import BehaviorLog, validate_consistency, compute_integrity_score
session = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
garp = validate_consistency(session)       # GARPResult with violation cycles, matrices
ccei = compute_integrity_score(session)    # AEIResult with binary search details
```

## 4 Choice Categories

```
                Test (bool)     Score (0-1)     Recover (vector)  Structure (bool)
Budget          GARP, WARP      CCEI, MPI, HM   Utility, CV/EV    HARP, Separability
Discrete        SARP, RUM LP    HM (menu)        Ordinal utility   Congruence
Production      Prod GARP       Prod CCEI        Tech efficiency   Cost minimization
Intertemporal   Exp discount    —                Discount delta    Quasi-hyperbolic
```

| Category | Input format | Example domain |
|----------|-------------|----------------|
| **Budget** | `(prices T×K, quantities T×K)` | E-commerce, grocery, food delivery |
| **Discrete** | `(menus, choices)` or `(menus, frequencies)` | Surveys, A/B tests, recommendations |
| **Production** | `(input_p, input_q, output_p, output_q)` | Supply chain, manufacturing |
| **Intertemporal** | `(amounts, dates, chosen)` | Subscriptions, savings, loyalty |

## Performance

The Rust engine (`rpt-core`) handles graph algorithms and LP solving via Rayon thread pool. Python handles I/O and the user-facing API.

| Users | Metrics | Time | Throughput |
|-------|---------|------|------------|
| 1,000 | 5 | 0.1s | 10,000/s |
| 10,000 | 5 | 2s | 5,000/s |
| 100,000 | 5 | 20s | 5,000/s |

18-100x faster than pure Python. Memory stays bounded via streaming chunks.

## Documentation

**Full docs**: https://prefgraph.readthedocs.io/ — examples, theory, API reference, and applications.

- Benchmarks overview: https://prefgraph.readthedocs.io/en/latest/benchmarks.html
- LLM Consistency Benchmarks: https://prefgraph.readthedocs.io/en/latest/budget/app_llm_benchmark.html
- E‑commerce benchmarks (incl. Amazon): https://prefgraph.readthedocs.io/en/latest/benchmarks_ecommerce.html
- Dataset loaders (15+): https://prefgraph.readthedocs.io/en/latest/api.html#dataset-loaders

## License

MIT
