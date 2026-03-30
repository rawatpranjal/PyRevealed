Budgets
=======

.. raw:: html

   <div style="margin: 2em 0; max-width: 600px; margin-left: auto; margin-right: auto; text-align: center;">
     <img src="../_static/budget_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
     <p class="gif-caption" style="margin-top: 10px; font-size: 0.9em; color: #555;"><strong>Budget choices.</strong> CCEI measures how much budgets must shrink to remove contradictions.</p>
   </div>

Budget analysis starts with a sequence of shopping trips. Each trip records the prices that were available and the quantities that were bought. From this, PrefGraph builds a directed graph where an edge from trip A to trip B means the bundle from B was affordable at A's prices but was not chosen.

GARP checks whether this graph contains preference cycles. CCEI measures how much budgets must shrink before the cycles disappear. MPI finds the costliest cycle. Houtman-Maks counts the fewest trips to drop to eliminate all cycles. Scores closer to 1 indicate behavior that is more consistent with utility maximization.

.. code-block:: python

   from prefgraph import BehaviorLog, validate_consistency, compute_integrity_score
   import numpy as np

   # 3 shopping trips, 2 goods — rows are observations, columns are goods
   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   # GARP: does a consistent utility function exist? (True/False)
   garp = validate_consistency(log)
   # CCEI: how much must budgets shrink to remove contradictions? (0–1)
   ccei = compute_integrity_score(log)
   print(f"GARP consistent: {garp.is_consistent}")
   print(f"CCEI: {ccei.efficiency_index:.4f}")

.. code-block:: text

   GARP consistent: False
   CCEI: 0.8750

``Engine.analyze_arrays()`` scores thousands of users in one Rust-backed batch call, running GARP, CCEI, MPI, HM, HARP, VEI, and a utility feasibility check. The per-user Functions API adds everything the Engine does not yet batch, including recovered utility vectors, welfare measurement, the Slutsky matrix, separability tests, and spatial preference recovery.

Theory
------

.. toctree::
   :maxdepth: 1

   theory_foundations
   theory_consistency
   theory_efficiency
   theory_structure

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorial
   tutorial_budget_advanced
   tutorial_ecommerce

Applications
------------

.. toctree::
   :maxdepth: 1

   app_grocery
   app_llm_alignment

Examples
--------

.. toctree::
   :maxdepth: 1

   examples
