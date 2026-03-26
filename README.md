# PyRevealed

Rationality scores for every user, at scale. Rust engine, Python interface.

[![PyPI](https://img.shields.io/pypi/v/pyrevealed)](https://pypi.org/project/pyrevealed/)
[![Documentation](https://readthedocs.org/projects/pyrevealed/badge/?version=latest)](https://pyrevealed.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

```bash
pip install pyrevealed
```

## What it does

Scores how consistently each user's choices align with rational utility maximization. Feed it choice data, get back per-user scores you can plug into any ML pipeline, segmentation model, or dashboard.

```python
from pyrevealed.engine import Engine
import numpy as np

# Each user: (prices T x K, quantities T x K)
users = [(prices_i, quantities_i) for prices_i, quantities_i in user_data]

engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
results = engine.analyze_arrays(users)

for r in results:
    print(r.is_garp, r.ccei, r.mpi, r.is_harp)
```

```
is_garp  ccei   mpi    is_harp  hm
False    0.847  0.023  True     18/20
True     1.000  0.000  True     20/20
False    0.791  0.041  False    15/20
```

## Scores

| Score | Field | What it measures | Range |
|-------|-------|-----------------|-------|
| Consistency | `is_garp` | Are choices rationalizable? (GARP) | bool |
| Efficiency | `ccei` | How close to perfectly rational? (Afriat) | 0-1 |
| Exploitability | `mpi` | Value left on the table per choice (Karp cycle) | 0-1 |
| Homotheticity | `is_harp` | Do preferences scale with budget? | bool |
| Noise fraction | `hm_consistent/hm_total` | Fraction of rationalizable choices (Houtman-Maks) | 0-1 |
| Utility recovery | `utility_success` | Can latent utility be reconstructed? (Afriat LP) | bool |
| Per-obs efficiency | `vei_mean` | Average efficiency across observations (Varian) | 0-1 |

Every score is a feature. Use them for fraud detection, user segmentation, A/B testing, churn prediction, or personalization.

## Performance

The Rust engine (`rpt-core`) handles graph algorithms and LP solving via Rayon thread pool. Python handles I/O and the user-facing API.

| Users | Metrics | Time | Throughput |
|-------|---------|------|------------|
| 1,000 | 5 | 0.1s | 10,000/s |
| 10,000 | 5 | 2s | 5,000/s |
| 100,000 | 5 | 20s | 5,000/s |

18-100x faster than pure Python. Memory stays bounded via streaming chunks.

## How it works

```
User choice data (prices + quantities per observation)
       |
       v
  +-----------+
  | Engine    |  partition by user, stream in chunks
  +-----+-----+
        |
        v
  +-----------+
  | Rust +    |  SCC decomposition -> Floyd-Warshall transitive closure
  | Rayon     |  Karp's cycle algorithm -> HiGHS LP solver
  +-----+-----+
        |
        v
  list[EngineResult]  (one per user)
```

## Core algorithms

| Algorithm | Module | Computation |
|-----------|--------|-------------|
| GARP | `garp.py` | Boolean cycle detection via SCC + Floyd-Warshall |
| CCEI (AEI) | `aei.py` | Binary search over efficiency levels |
| MPI | `mpi.py` | Karp's max-mean-weight cycle O(T^3) |
| HARP | `harp.py` | Max-product cycle in log-space |
| Houtman-Maks | `mpi.py` | Greedy feedback vertex set / ILP |
| Quasilinear | `quasilinear.py` | Bellman-Ford negative cycle detection |
| Utility recovery | `utility.py` | Afriat LP via HiGHS |
| VEI | `vei.py` | Per-observation efficiency LP |
| Menu SARP | `abstract_choice.py` | Cycle detection on item-space graph |
| Attention | `attention.py` | Consideration sets + graph + LP |
| Production GARP | `production.py` | Profit comparison graph |

## 5 Choice Categories

We test whether observed choices are consistent with rational optimization — without estimating parameters. We answer "does a rational model exist?", not "which model is it?".

| Category | Input format | Example domain | What we test |
|----------|-------------|----------------|-------------|
| **Budget** | `(prices T×K, quantities T×K)` | E-commerce, grocery, delivery | GARP, CCEI, MPI, HARP, utility recovery |
| **Menu** | `(menus: list[set], choices: list[int])` | Surveys, recommendations, UI clicks | SARP, WARP, attention |
| **Stochastic** | `(menus: list[set], frequencies: list[dict])` | A/B tests, experimental panels | RUM consistency LP, regularity |
| **Production** | `(input_p, input_q, output_p, output_q)` | Supply chain, manufacturing | Profit maximization GARP |
| **Intertemporal** | `(amounts, dates, chosen)` | Subscription, savings, loyalty | Discount factor bounds |

Each user is a tuple of arrays. For batch scoring, pass a list of tuples:

```python
users = [
    (prices_user_0, quantities_user_0),  # T0 x K arrays
    (prices_user_1, quantities_user_1),  # T1 x K arrays (T can vary per user)
    ...
]
results = engine.analyze_arrays(users)
```

## Documentation

**[Full docs](https://pyrevealed.readthedocs.io/)**

## License

MIT
