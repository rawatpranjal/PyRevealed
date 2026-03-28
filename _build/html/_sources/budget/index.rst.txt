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
