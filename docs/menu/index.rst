Menus
=====

.. raw:: html

   <div style="margin: 2em 0; max-width: 600px; margin-left: auto; margin-right: auto; text-align: center;">
     <img src="../_static/menu_hero.gif" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" alt="Menu Choices">
     <p class="gif-caption" style="margin-top: 10px; font-size: 0.9em; color: #555;"><strong>Menu choices.</strong> HM counts how many choices to discard to restore consistency.</p>
   </div>

Menu analysis tests whether choices from finite sets follow a stable ranking. There are no prices or budgets. The input is a sequence of menus paired with whichever option was picked from each one.

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

.. code-block:: python

   from prefgraph import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency

   # 4 observations: each menu is a set of item indices, choice is which was picked
   log = MenuChoiceLog(
       menus=[
           frozenset({0, 1, 2}),   # menu 1: {Pizza, Burger, Salad}
           frozenset({1, 2, 3}),   # menu 2: {Burger, Salad, Pasta}
           frozenset({0, 3}),      # menu 3: {Pizza, Pasta}
           frozenset({0, 1, 3}),   # menu 4: {Pizza, Burger, Pasta}
       ],
       choices=[0, 1, 0, 0],      # picked Pizza, Burger, Pizza, Pizza
       item_labels=["Pizza", "Burger", "Salad", "Pasta"],
   )
   # SARP: are there any preference cycles? (stricter than WARP)
   sarp = validate_menu_sarp(log)
   # HM: fraction of choices consistent with a single ranking
   hm = compute_menu_efficiency(log)
   print(f"SARP consistent: {sarp.is_consistent}")
   print(f"HM efficiency: {hm.efficiency_index:.2f}")

.. code-block:: text

   SARP consistent: True
   HM efficiency: 1.00

Deterministic data feeds directly into ``Engine.analyze_menus()`` for batch Rust processing. Stochastic and risk data use the per-user Functions API because their inputs do not map to the tuple format the batch engine expects. The axioms and scores are identical across paths.

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
   app_finn_slates

Examples
--------

.. toctree::
   :maxdepth: 1

   examples
