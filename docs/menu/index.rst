Menus
=====

Menu-based analysis evaluates consistency when choices come from finite
sets of alternatives without prices. Covers classical WARP/SARP,
limited attention, random utility, and risk preferences.

.. raw:: html

   <div style="margin: 2em 0; max-width: 600px; margin-left: auto; margin-right: auto; text-align: center;">
     <img src="../_static/menu_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
     <p class="gif-caption" style="margin-top: 10px; font-size: 0.9em; color: #555;"><strong>Menu choices.</strong> HM counts how many choices to discard to restore consistency.</p>
   </div>

.. list-table::
   :widths: 33 34 33
   :align: center
   :class: gif-grid

   * - .. image:: ../_static/deterministic.gif
          :alt: Deterministic logic
          :width: 100%
     - .. image:: ../_static/stochastic.gif
          :alt: Stochastic logic
          :width: 100%
     - .. image:: ../_static/risk.gif
          :alt: Risk logic
          :width: 100%

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

Deterministic data feeds directly into ``Engine.analyze_menus()`` for batch Rust processing. Stochastic and Risk data use the per-user Functions API (``fit_luce_model``, ``compute_risk_profile``, etc.) because their inputs do not map to the simple tuple format the batch engine expects. The axioms and scores are identical across paths; only the entry point differs.

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency

   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),   # chose Pizza from {Pizza, Burger, Salad}
           frozenset({1, 2, 3}),   # chose Burger from {Burger, Salad, Pasta}
           frozenset({0, 3}),      # chose Pizza from {Pizza, Pasta}
           frozenset({0, 1, 3}),   # chose Pizza from {Pizza, Burger, Pasta}
       ],
       choices=[0, 1, 0, 0],
       item_labels=["Pizza", "Burger", "Salad", "Pasta"],
   )
   sarp = validate_menu_sarp(log)
   hm = compute_menu_efficiency(log)
   print(f"SARP consistent: {sarp.is_consistent}")
   print(f"HM efficiency: {hm.efficiency_index:.2f}")

.. code-block:: text

   SARP consistent: True
   HM efficiency: 1.00

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
   tutorial_stochastic
   tutorial_attention

Applications
------------

.. toctree::
   :maxdepth: 1

   app_recsys

Examples
--------

.. toctree::
   :maxdepth: 1

   examples
