Quickstart
==========

Install
-------

.. code-block:: bash

   pip install pyrevealed

Score 100 Users in 5 Lines
--------------------------

.. code-block:: python

   from pyrevealed.datasets import load_demo
   from pyrevealed.engine import Engine

   users = load_demo()  # 100 synthetic consumers, no download
   engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
   results = engine.analyze_arrays(users)

   for r in results[:5]:
       print(r)

.. code-block:: text

   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (25us)
   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (19us)
   EngineResult: [-] 2 violations  ccei=0.9920  mpi=0.0084  hm=14/15  (32us)
   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (20us)
   EngineResult: [+] GARP-consistent  ccei=1.0000  hm=15/15  (23us)

Read the Results
----------------

Each ``EngineResult`` is one user's rationality profile:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Range
     - Meaning
   * - ``is_garp``
     - bool
     - Are choices rationalizable? True = consistent with utility maximization.
   * - ``ccei``
     - (0, 1]
     - Afriat efficiency. 1.0 = perfectly rational. Lower = more waste.
   * - ``mpi``
     - [0, 1)
     - Money Pump Index. 0.0 = unexploitable. Higher = more inconsistent.
   * - ``hm_consistent / hm_total``
     - [0, 1]
     - Houtman-Maks fraction of rationalizable observations.

Call ``r.summary()`` for a formatted report, or ``r.to_dict()`` for serialization.

Which API?
----------

PyRevealed has two APIs. Use whichever fits your task:

**Engine** — batch scoring (thousands of users, Rust backend):

.. code-block:: python

   from pyrevealed.engine import Engine
   from pyrevealed.datasets import load_demo

   results = Engine(metrics=["garp", "ccei", "mpi"]).analyze_arrays(load_demo())
   # list[EngineResult] — flat scores, ready for pandas

**Function API** — deep single-user analysis (violation details, preference graphs):

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
   import numpy as np

   prices = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
   quantities = np.array([[4.0, 1.0], [1.0, 4.0], [2.0, 2.0]])
   session = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   garp = validate_consistency(session)     # GARPResult: violations, preference matrices
   ccei = compute_integrity_score(session)  # AEIResult: binary search details

Your Own Data
-------------

**Fastest path** — ``analyze()`` takes a DataFrame and returns a DataFrame:

.. code-block:: python

   import pyrevealed as rp

   # Wide format (one row per observation, items as columns)
   results = rp.analyze(df, user_col="user_id",
                        cost_cols=["price_A", "price_B"],
                        action_cols=["qty_A", "qty_B"])

   # Long format (transaction logs — one row per item per time)
   results = rp.analyze(df, user_col="user_id", item_col="product",
                        cost_col="price", action_col="quantity", time_col="week")

   # Menu/click data
   results = rp.analyze(df, user_col="user_id",
                        menu_col="shown_items", choice_col="clicked")

   print(results[["is_garp", "ccei", "mpi"]].describe())

Format is auto-detected from which parameters you provide. Default metrics: ``garp``, ``ccei``, ``mpi``.
Customize with ``metrics=["garp", "ccei", "mpi", "hm", "harp"]``.

**Power-user path** — ``BehaviorPanel`` for full control:

.. code-block:: python

   from pyrevealed import BehaviorPanel
   from pyrevealed.engine import Engine

   panel = BehaviorPanel.from_dataframe(df, user_col="user_id",
                                        cost_cols=["price_A", "price_B"],
                                        action_cols=["qty_A", "qty_B"])

   engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
   results = engine.analyze_arrays(panel.to_engine_tuples())

Next Steps
----------

- :doc:`budget/tutorial` — budget choice analysis in depth
- :doc:`menu/tutorial_menu_choice` — menu/discrete choice analysis
- :doc:`api` — full API reference
- :doc:`performance` — benchmarks and scaling
