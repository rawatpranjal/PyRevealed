Menu Choice
===========

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

   from pyrevealed import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency

   log = MenuChoiceLog(menus=menus, choices=choices)
   sarp = validate_menu_sarp(log)            # Test: bool
   hm   = compute_menu_efficiency(log)       # Score: 0→1

Theory
------

.. toctree::
   :maxdepth: 1

   theory_abstract
   theory_attention
   theory_stochastic

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorial_menu_choice
   tutorial_attention
   tutorial_stochastic
   tutorial_risk
   tutorial_context_effects
   tutorial_ranking

Applications
------------

.. toctree::
   :maxdepth: 1

   app_recsys
