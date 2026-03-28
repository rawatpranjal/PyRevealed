Budgets
=======

**Input:** ``BehaviorLog`` — prices × quantities (continuous budget sets)

Every purchase at observed prices adds directed edges to the agent's **observation graph** (nodes = shopping trips, edges = revealed preferences). Budget analysis checks whether this graph is acyclic (GARP), scores how close it is (CCEI, MPI), and recovers utility.
The classical setting of Samuelson (1938), Afriat (1967), and Varian (1982).

.. admonition:: What can you do?

   - **Test**: GARP, WARP, SARP, HARP, GAPP, integrability, separability, gross substitutes
   - **Score**: CCEI, MPI, VEI, Houtman-Maks, Swaps, Bronars power
   - **Recover**: Utility, demand, welfare (CV/EV), Slutsky matrix, expenditure function
   - **Structure**: Separability partitions, quasilinear, additive, spatial

.. code-block:: python

   from prefgraph import BehaviorLog, validate_consistency, compute_integrity_score

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
   garp = validate_consistency(log)          # Test: bool
   ccei = compute_integrity_score(log)       # Score: 0→1

When To Use
-----------

- Use budgets when you have prices and purchased quantities per observation (e.g., weekly baskets by category).
- If you only know which item was chosen from a set (no prices), use :doc:`/menu/index` instead.

Data Shape and Types
--------------------

- ``BehaviorLog(cost_vectors, action_vectors)`` expects two 2D arrays of shape ``(T, N)``
  - ``T`` = number of observations (trips, weeks, orders)
  - ``N`` = number of goods/features (categories, aisles)
- All values must be finite. Costs must be strictly positive; quantities non‑negative.
- If you have a pandas DataFrame, see ``BehaviorPanel.from_dataframe`` to build many logs at once.

Outputs You Get
---------------

- ``validate_consistency`` → pass/fail (GARP, with violation details in the result object)
- ``compute_integrity_score`` (Afriat/CCEI) → number in [0, 1]; higher = more consistent
- ``compute_confusion_metric`` (MPI) → exploitability in [0, 1]; higher = more inconsistent
- ``recover_utility`` → Afriat utility values if data is GARP‑consistent

Interpretation Guide
--------------------

- CCEI (0→1):
  - ~0.95–1.00: near‑perfect; small local contradictions only
  - ~0.80–0.95: moderate inconsistencies; investigate specific observations
  - <0.80: substantial waste/violations; expect low predictive stability
- MPI (0→1):
  - 0.00: no money‑pump cycles (ideal)
  - 0.05–0.20: mild to moderate arbitrage potential
  - >0.20: large cycles; consider segmentation or filtering

Batch Mode (Many Users)
-----------------------

.. code-block:: python

   from prefgraph.engine import Engine
   # users = list of (prices, quantities) tuples, each shaped (T, N)
   engine = Engine(metrics=["garp", "ccei", "mpi", "harp"])  # Rust backend if available
   results = engine.analyze_arrays(users)
   # results[i]: EngineResult with is_garp, ccei, mpi, is_harp, hm_consistent/total, etc.

Common Pitfalls
---------------

- Zero or negative prices: filter or impute; required to be strictly > 0.
- Mismatched shapes: ``cost_vectors`` and ``action_vectors`` must both be ``(T, N)``.
- Aggregation drift: keep the same set of goods across observations; reindex missing goods to 0 quantity.
- Sparse/short histories (very small ``T``): expect low test power; see ``compute_test_power``.

Theory
------

.. toctree::
   :maxdepth: 1

   theory_foundations
   theory_consistency
   theory_efficiency
   theory_structure
   theory_advanced
   theory_spatial

Examples
--------

.. toctree::
   :maxdepth: 1

   examples
