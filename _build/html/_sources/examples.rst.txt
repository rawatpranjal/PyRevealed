Examples
========

.. contents:: On this page
   :local:
   :depth: 1

Batch Scoring (Engine)
----------------------

Score many users in parallel. The Engine auto-selects Rust or Python backend.

.. code-block:: python

   from prefgraph.engine import Engine
   import numpy as np

   rng = np.random.RandomState(42)
   users = [
       (rng.rand(20, 5).astype(np.float64) + 0.1,
        rng.rand(20, 5).astype(np.float64) + 0.1)
       for _ in range(10)
   ]

   engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
   results = engine.analyze_arrays(users)

   for i, r in enumerate(results[:5]):
       hm = f"{r.hm_consistent}/{r.hm_total}"
       print(f"user {i}  garp={str(r.is_garp):5s}  ccei={r.ccei:.3f}"
             f"  mpi={r.mpi:.3f}  harp={str(r.is_harp):5s}  hm={hm}")

.. code-block:: text

   user 0  garp=False  ccei=0.755  mpi=0.285  harp=False  hm=13/20
   user 1  garp=False  ccei=0.799  mpi=0.241  harp=False  hm=15/20
   user 2  garp=False  ccei=0.910  mpi=0.257  harp=False  hm=16/20
   user 3  garp=False  ccei=0.785  mpi=0.318  harp=False  hm=14/20
   user 4  garp=False  ccei=0.716  mpi=0.356  harp=False  hm=13/20

Each ``EngineResult`` has: ``is_garp``, ``ccei``, ``mpi``, ``is_harp``,
``hm_consistent``, ``hm_total``, ``utility_success``, ``vei_mean``, ``vei_min``.

Single-User Analysis
--------------------

Individual functions for one user at a time. Each returns a detailed result object.

.. code-block:: python

   from prefgraph import BehaviorLog, check_garp, compute_aei, compute_mpi
   import numpy as np

   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   garp = check_garp(log)
   print(f"GARP consistent: {garp.is_consistent}")
   print(f"Violations: {len(garp.violations)}")
   print(f"R matrix shape: {garp.direct_revealed_preference.shape}")

   aei = compute_aei(log, method="discrete")
   print(f"CCEI: {aei.efficiency_index:.4f}")
   print(f"Iterations: {aei.binary_search_iterations}")

   mpi = compute_mpi(log)
   print(f"MPI: {mpi.mpi_value:.4f}")
   print(f"Worst cycle: {mpi.worst_cycle}")

.. code-block:: text

   GARP consistent: False
   Violations: 3
   R matrix shape: (3, 3)
   CCEI: 0.8750
   Iterations: 2
   MPI: 0.1250
   Worst cycle: (0, 1, 0)

Full Report
-----------

``BehavioralAuditor.summary()`` produces a statsmodels-style report combining
all consistency tests, efficiency metrics, and interpretation:

.. code-block:: python

   from prefgraph import BehaviorLog, BehavioralAuditor
   import numpy as np

   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   auditor = BehavioralAuditor()
   print(auditor.summary(log))

.. code-block:: text

   ======================================================================
                             BEHAVIORAL SUMMARY
   ======================================================================
   User ID: N/A                       GARP: [-] FAIL
   No. Observations: 3                WARP: [-] FAIL
   No. Goods: 2                       SARP: [-] FAIL
   Method: Floyd-Warshall             AEI: 0.8750
   Computation Time: 5.39 ms          MPI: 0.1250
   ======================================================================

   Input Data:
   ----------------------------------------------------------------------
                           mean   std dev       min       max
     Prices               1.500     0.408     1.000     2.000
     Quantities           2.500     0.408     2.000     3.000
     Expenditure          7.833     0.236     7.500     8.000

   Revealed Preference Graph:
   ----------------------------------------------------------------------
     R  (direct, p'x >= p'y) ..................... 9 / 9 edges (100.0%)
     P  (strict, p'x >  p'y) ...................... 4 / 9 edges (44.4%)
     R* (transitive closure) ..................... 9 / 9 edges (100.0%)
     Violation pairs (R* & P') ...................................... 4

   Consistency Tests:
   ----------------------------------------------------------------------
     GARP ......................................... [-] FAIL (3 cycles)
     WARP ...................................... [-] FAIL (1 violation)
     SARP ......................................... [-] FAIL (3 cycles)

   Goodness-of-Fit:
   ----------------------------------------------------------------------
     Afriat Efficiency (AEI) ................................... 0.8750
       Binary search iterations ..................................... 2
       Budget waste ............................................ 12.50%
     Money Pump Index (MPI) .................................... 0.1250
       Violation cycles ............................................. 3
       Worst cycle cost ........................................ 0.1250
       Total expenditure ....................................... $23.50
     Houtman-Maks Index ........................................ 0.3333
       Observations removed ..................................... 2 / 3

   Interpretation:
   ----------------------------------------------------------------------
     Moderate consistency - some behavioral anomalies present
     ~12.5% budget waste; an arbitrager could extract ~12.5%.

HARP (Homotheticity)
--------------------

Test whether preferences scale proportionally with income:

.. code-block:: python

   from prefgraph import BehaviorLog
   from prefgraph.algorithms.harp import check_harp
   import numpy as np

   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   harp = check_harp(log)
   print(f"Homothetic: {harp.is_consistent}")

.. code-block:: text

   Homothetic: False

VEI (Per-Observation Efficiency)
--------------------------------

Unlike CCEI which gives one global score, VEI identifies which specific
observations are problematic:

.. code-block:: python

   from prefgraph import BehaviorLog
   from prefgraph.algorithms.vei import compute_vei
   import numpy as np

   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   vei = compute_vei(log)
   print(f"Mean efficiency: {vei.mean_efficiency:.4f}")
   print(f"Min efficiency: {vei.min_efficiency:.4f}")
   print(f"Worst observation: {vei.worst_observation}")
   for i, e in enumerate(vei.efficiency_vector):
       print(f"  obs {i}: {e:.3f}")

.. code-block:: text

   Mean efficiency: 1.0000
   Min efficiency: 1.0000
   Worst observation: 0
     obs 0: 1.000
     obs 1: 1.000
     obs 2: 1.000

Utility Recovery
----------------

If data is GARP-consistent, recover latent utility values via Afriat's LP:

.. code-block:: python

   from prefgraph import BehaviorLog, recover_utility
   import numpy as np

   # Consistent data (GARP passes)
   prices = np.array([[1.0, 2.0], [2.0, 1.0]])
   quantities = np.array([[4.0, 1.0], [1.0, 4.0]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   result = recover_utility(log)
   print(f"Success: {result.success}")
   print(f"Utility values: {result.utility_values}")
   print(f"Lagrange multipliers: {result.lagrange_multipliers}")

.. code-block:: text

   Success: True
   Utility values: [0. 0.]
   Lagrange multipliers: [1.e-06 1.e-06]

Quasilinear Utility
-------------------

Test whether utility has the form U(x, m) = v(x) + m (no income effects):

.. code-block:: python

   from prefgraph import BehaviorLog
   from prefgraph.algorithms.quasilinear import check_quasilinearity
   import numpy as np

   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   ql = check_quasilinearity(log)
   print(f"Quasilinear: {ql.is_quasilinear}")
   print(f"Violations: {len(ql.violations)}")

.. code-block:: text

   Quasilinear: False
   Violations: 9

Menu Choices (SARP)
-------------------

For discrete choices from menus (no prices). Tests whether item preferences
form a consistent ranking:

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency
   import numpy as np

   menus = [
       frozenset({0, 1, 2}),  # chose 0 from {0, 1, 2}
       frozenset({1, 2, 3}),  # chose 1 from {1, 2, 3}
       frozenset({0, 2, 3}),  # chose 2 from {0, 2, 3}
       frozenset({0, 1}),     # chose 1 from {0, 1}
       frozenset({2, 3}),     # chose 3 from {2, 3}
   ]
   choices = [0, 1, 2, 1, 3]

   log = MenuChoiceLog(menus=menus, choices=choices)

   sarp = validate_menu_sarp(log)
   print(f"SARP consistent: {sarp.is_consistent}")
   print(f"Violations: {sarp.num_violations}")

   hm = compute_menu_efficiency(log)
   print(f"Observations to remove: {len(hm.removed_observations)}/{len(menus)}")

.. code-block:: text

   SARP consistent: False
   Violations: 6
   Observations to remove: 2/5

Production Data
---------------

Test whether a firm's input/output decisions are consistent with profit
maximization:

.. code-block:: python

   from prefgraph import ProductionLog
   from prefgraph.algorithms.production import test_profit_maximization
   import numpy as np

   rng = np.random.RandomState(42)
   log = ProductionLog(
       input_prices=rng.rand(10, 3) + 0.5,
       input_quantities=rng.rand(10, 3) + 0.1,
       output_prices=rng.rand(10, 2) + 1.0,
       output_quantities=rng.rand(10, 2) + 0.1,
   )

   result = test_profit_maximization(log)
   print(f"Profit maximizing: {result.is_consistent}")
   print(f"Violations: {result.num_violations}")

.. code-block:: text

   Profit maximizing: False
   Violations: 15

Loading Data
------------

From a pandas DataFrame:

.. code-block:: python

   from prefgraph import BehaviorPanel

   panel = BehaviorPanel.from_dataframe(
       df,
       user_col="customer_id",
       cost_cols=["price_a", "price_b", "price_c"],
       action_cols=["qty_a", "qty_b", "qty_c"],
   )

From individual logs:

.. code-block:: python

   from prefgraph import BehaviorLog, BehaviorPanel
   import numpy as np

   logs = [
       BehaviorLog(prices, quantities, user_id=f"user_{i}")
       for i, (prices, quantities) in enumerate(user_data)
   ]
   panel = BehaviorPanel.from_logs(logs)

Built-in datasets (require separate download):

.. code-block:: python

   from prefgraph.datasets import load_dunnhumby, load_open_ecommerce, load_uci_retail

   panel = load_dunnhumby()           # 2,500 households, 10 goods, 104 weeks
   panel = load_open_ecommerce()      # 4,700 consumers, 50 categories, 66 months
   panel = load_uci_retail()          # 1,800 customers, 50 products, 13 months
