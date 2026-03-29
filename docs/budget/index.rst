Budgets
=======

**Input:** ``BehaviorLog`` - prices × quantities (continuous budget sets)

.. raw:: html

   <div style="margin: 2em 0; max-width: 600px; margin-left: auto; margin-right: auto; text-align: center;">
     <img src="../_static/budget_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Budget Choices">
     <p class="gif-caption" style="margin-top: 10px; font-size: 0.9em; color: #555;"><strong>Budget choices.</strong> CCEI measures how much budgets must shrink to remove contradictions.</p>
   </div>

Every purchase at observed prices adds directed edges to the agent's **observation graph** (nodes = shopping trips, edges = revealed preferences). Budget analysis checks whether this graph is acyclic (GARP), scores how close it is (CCEI, MPI), and recovers utility.
The classical setting of Samuelson (1938), Afriat (1967), and Varian (1982).

``Engine.analyze_arrays()`` scores thousands of users in one Rust-backed batch call, running GARP, CCEI, MPI, HM, HARP, VEI, and a utility feasibility check. The per-user Functions API adds everything the Engine does not yet batch: recovered utility vectors, welfare measurement (CV/EV), the Slutsky matrix, separability tests, and spatial preference recovery.

.. code-block:: python

   from prefgraph import BehaviorLog, validate_consistency, compute_integrity_score
   import numpy as np

   prices = np.array([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
   quantities = np.array([[3.0, 2.0], [2.0, 3.0], [2.5, 2.5]])
   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

   garp = validate_consistency(log)
   ccei = compute_integrity_score(log)
   print(f"GARP consistent: {garp.is_consistent}")
   print(f"CCEI: {ccei.efficiency_index:.4f}")

.. code-block:: text

   GARP consistent: False
   CCEI: 0.8750

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
