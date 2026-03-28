Menus
=====

Menu-based analysis evaluates consistency when choices come from finite
sets of alternatives without prices. Covers classical WARP/SARP,
limited attention, random utility, and risk preferences.

**3 data subtypes:**

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Subtype
     - Input Class
     - Description
   * - **Deterministic**
     - ``MenuChoiceLog``
     - menus → single choices (e.g., which product was clicked)
   * - **Stochastic**
     - ``StochasticChoiceLog``
     - menus → choice frequencies (e.g., 60% A, 30% B, 10% C)
   * - **Risk / Lotteries**
     - ``RiskChoiceLog``
     - lotteries → choices (e.g., gamble A vs gamble B)

.. admonition:: What can you do?

   - **Test**: WARP, SARP, Congruence, WARP-LA, RAM, RUM, IIA, regularity, expected utility
   - **Score**: Menu HM, distance to RUM, predictive success
   - **Recover**: Ordinal utility, consideration sets, attention probabilities, Luce/RUM, risk profile
   - **Structure**: Attention bounds, RAM parameters

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency

   log = MenuChoiceLog(menus=menus, choices=choices)
   sarp = validate_menu_sarp(log)            # Test: bool
   hm   = compute_menu_efficiency(log)       # Score: 0→1

When To Use
-----------

- Use menus when each observation is a set of available items (a menu) and a single chosen item.
- If you observe prices and purchased quantities instead, use :doc:`/budget/index`.

Data Shape and Types
--------------------

- Deterministic: ``MenuChoiceLog(menus, choices)``
  - ``menus``: iterable of sets/frozensets of item indices (e.g., ``{0, 3, 7}``)
  - ``choices``: iterable of integers, each contained in the corresponding menu
- Stochastic: ``StochasticChoiceLog(menus, frequencies)``
  - ``frequencies``: per‑menu frequency vectors that sum to 1
- Risk: ``RiskChoiceLog(lotteries, choices)`` where items are lotteries
- Items must be indexed by integers [0..K‑1]. For string IDs, map to ints per user.

Outputs You Get
---------------

- ``validate_menu_warp``/``validate_menu_sarp`` → pass/fail with violation details
- ``compute_menu_efficiency`` (HM) → fraction of observations to remove (higher = better)
- ``fit_menu_preferences`` → ordinal utility/ranking if consistent

Interpretation Guide
--------------------

- SARP pass: there exists a strict ranking that explains all choices across menus.
- HM efficiency (0→1): fraction of observations kept after removing minimal contradictions.
  - 1.00: perfectly consistent
  - 0.85–0.99: minor flips; often due to near‑ties or exposure noise
  - <0.85: systematic context effects or instability
- IIA/Regularity: compare pair results with triples; violations indicate menu dependence.

Batch Mode (Many Users)
-----------------------

.. code-block:: python

   from prefgraph.engine import Engine
   # users = list of (menus, choices) tuples per user
   engine = Engine(metrics=["sarp", "warp", "hm"])  # menu metrics are auto‑computed
   results = engine.analyze_menus(users)  # returns list[MenuResult]
   # results[i]: MenuResult with is_sarp, is_warp, hm_consistent/total, etc.

Common Pitfalls
---------------

- Menus containing items not seen before or inconsistent indexing across observations.
- Choices not contained in the menu for that observation.
- Over‑aggregated sessions: if a session contains multiple purchases, split into multiple menus.
- Very small menus (always size 1): no information about preferences; filter these out.

Theory
------

.. toctree::
   :maxdepth: 1

   theory_abstract
   theory_attention
   theory_stochastic

Examples
--------

.. toctree::
   :maxdepth: 1

   examples
