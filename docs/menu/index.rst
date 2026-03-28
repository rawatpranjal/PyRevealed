Menus
=====

Menu-based analysis evaluates consistency when choices come from finite
sets of alternatives without prices. Covers classical WARP/SARP,
limited attention, random utility, and risk preferences.

**3 data subtypes:**

.. raw:: html

   <div style="display: flex; gap: 15px; margin-bottom: 20px;">
       <div style="flex: 1;"><img src="../_static/deterministic.gif" alt="Deterministic logic" style="width: 100%; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" /></div>
       <div style="flex: 1;"><img src="../_static/stochastic.gif" alt="Stochastic logic" style="width: 100%; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" /></div>
       <div style="flex: 1;"><img src="../_static/risk.gif" alt="Risk logic" style="width: 100%; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" /></div>
   </div>

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
