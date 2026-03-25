Quick Start
===========

PyRevealed scores how consistently each user's choices align with rational
utility maximization. The primary workflow: prepare user data, run the Engine,
read per-user scores.

1. Batch Scoring with Engine
----------------------------

Score many users in one call (uses Rust backend if installed):

.. code-block:: python

   from pyrevealed.engine import Engine
   import numpy as np

   # Each user: (prices T x K, quantities T x K)
   users = [
       (np.random.rand(20, 5) + 0.1, np.random.rand(20, 5) + 0.1)
       for _ in range(1000)
   ]

   engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
   results = engine.analyze_arrays(users)

   # Each result has: is_garp, ccei, mpi, is_harp, hm_consistent, hm_total
   for r in results[:3]:
       print(f"GARP={r.is_garp}  CCEI={r.ccei:.3f}  MPI={r.mpi:.3f}")

2. Load Panel Data
------------------

From a built-in dataset:

.. code-block:: python

   from pyrevealed.datasets import load_dunnhumby

   panel = load_dunnhumby(n_households=100)
   print(panel)  # BehaviorPanel(users=85, total_obs=4174)

From a pandas DataFrame:

.. code-block:: python

   from pyrevealed import BehaviorPanel

   panel = BehaviorPanel.from_dataframe(
       df,
       user_col="customer_id",
       cost_cols=["price_a", "price_b", "price_c"],
       action_cols=["qty_a", "qty_b", "qty_c"],
   )

From individual logs:

.. code-block:: python

   from pyrevealed import BehaviorLog, BehaviorPanel
   import numpy as np

   logs = [
       BehaviorLog(prices_array, quantities_array, user_id=f"user_{i}")
       for i, (prices_array, quantities_array) in enumerate(user_data)
   ]
   panel = BehaviorPanel.from_logs(logs)

2. Run Analysis
---------------

.. code-block:: python

   report = panel.summary()
   print(report)

This runs GARP, AEI, MPI, WARP, SARP, and Houtman-Maks for every user and
produces an aggregate ``PanelSummary`` with consistency rates, efficiency
distributions, and the most inconsistent users.

3. Read the Report
------------------

The panel summary shows:

- **Consistency Rates** --- what fraction of users pass GARP/WARP/SARP
- **Efficiency Distribution** --- mean, std, percentiles of AEI/MPI/HM across users
- **Most Inconsistent** --- bottom 5 users ranked by AEI
- **Temporal Breakdown** --- per-period stats (if panel has period structure)

4. Drill Down
--------------

Inspect a single user:

.. code-block:: python

   user_report = panel.analyze_user("household_1")
   print(user_report)

This produces a detailed ``BehavioralSummary`` with input data statistics,
revealed preference graph density, sub-metrics, and interpretation.

.. code-block:: text

   ======================================================================
                             BEHAVIORAL SUMMARY
   ======================================================================
   User ID: household_1               GARP: [-] FAIL
   No. Observations: 60               WARP: [-] FAIL
   No. Goods: 10                      SARP: [-] FAIL
   Method: Floyd-Warshall             AEI: 0.8542
   Computation Time: 213.88 ms        MPI: 0.2354
   ======================================================================

   Input Data:
   ----------------------------------------------------------------------
                           mean   std dev       min       max
     Prices               2.306     1.036     0.790     5.110
     Quantities           0.773     1.138     0.000     7.000
     Expenditure         16.215     8.005     1.990    39.300
   ...

5. Single-User Mode
-------------------

If you just need to analyze one user without a panel:

.. code-block:: python

   from pyrevealed import BehaviorLog, BehavioralSummary
   import numpy as np

   log = BehaviorLog(
       cost_vectors=np.array([[5, 4], [4, 5], [3, 6]]),
       action_vectors=np.array([[2, 1], [1, 2], [2, 2]]),
   )
   print(BehavioralSummary.from_log(log))

Available Datasets
------------------

.. code-block:: python

   from pyrevealed.datasets import load_dunnhumby, load_open_ecommerce, load_uci_retail

   panel = load_dunnhumby()           # 2,500 households, 10 goods, 104 weeks
   panel = load_open_ecommerce()      # 4,700 consumers, 50 categories, 66 months
   panel = load_uci_retail()          # 1,800 customers, 50 products, 13 months

Data must be downloaded separately. Each loader prints instructions if files are not found.
