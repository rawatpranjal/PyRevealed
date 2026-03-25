Examples
========

Core examples showing each data type and analysis pattern.

.. contents:: On this page
   :local:
   :depth: 1

Budget Panel Analysis
---------------------

The primary use case: analyze consistency across many users over time.

.. code-block:: python

   from pyrevealed.datasets import load_dunnhumby

   # Load 2,500 households x 10 grocery categories x 104 weeks
   panel = load_dunnhumby()
   report = panel.summary()
   print(report)

   # Drill into a specific user
   user_report = panel.analyze_user("household_42")
   print(user_report)

   # Filter to users with enough data
   big_panel = panel.filter(lambda log: log.num_observations >= 30)
   print(big_panel.summary())

**Temporal analysis** --- split by month to track consistency over time:

.. code-block:: python

   panel = load_dunnhumby(n_households=50, period="month")
   report = panel.summary()
   print(report)  # Shows Temporal Breakdown section

**From your own data** --- load from a DataFrame:

.. code-block:: python

   import pandas as pd
   from pyrevealed import BehaviorPanel

   df = pd.read_csv("transactions.csv")
   panel = BehaviorPanel.from_dataframe(
       df,
       user_col="customer_id",
       cost_cols=["price_milk", "price_bread", "price_eggs"],
       action_cols=["qty_milk", "qty_bread", "qty_eggs"],
       period_col="month",  # optional: enables temporal breakdown
   )
   print(panel.summary())

Menu-Based Choice
-----------------

Discrete choices from menus without prices. For surveys, experiments, voting.

.. code-block:: python

   from pyrevealed import MenuChoiceLog, MenuChoiceSummary

   # Restaurant choices: 6 visits, 4 dishes
   menus = [
       frozenset({0, 1, 2, 3}),  # Full menu
       frozenset({0, 1, 2}),     # No fish
       frozenset({0, 2, 3}),     # No steak
       frozenset({1, 2, 3}),     # No pasta
       frozenset({0, 1}),        # Pasta or steak only
       frozenset({2, 3}),        # Salad or fish only
   ]
   choices = [1, 1, 0, 1, 1, 3]  # Prefers steak > pasta > fish > salad

   log = MenuChoiceLog(menus, choices, item_labels=["Pasta", "Steak", "Salad", "Fish"])
   print(MenuChoiceSummary.from_log(log))

.. code-block:: text

   ======================================================================
                            MENU CHOICE SUMMARY
   ======================================================================
   No. Observations: 6                WARP: [+] PASS
   No. Alternatives: 4                SARP: [+] PASS
   Computation Time: 142.29 ms        Congruence: [+] PASS
   ======================================================================

   Input Data:
   ----------------------------------------------------------------------
                           mean   std dev       min       max
     Menu Size            2.833     0.687     2.000     4.000

     Unique Items Chosen ........................................ 3 / 4
     Choice Diversity .......................................... 0.7500

   Consistency Tests:
   ----------------------------------------------------------------------
     WARP .................................................... [+] PASS
     SARP .................................................... [+] PASS
     Congruence .............................................. [+] PASS

   Goodness-of-Fit:
   ----------------------------------------------------------------------
     Houtman-Maks Efficiency ................................... 1.0000
       Observations removed ..................................... 0 / 6

   Recovered Preference Order:
   ----------------------------------------------------------------------
     1 > 0 > 3 > 2

   Interpretation:
   ----------------------------------------------------------------------
     Choices are fully rationalizable by a complete preference ordering.
     Efficiency: 100.0% of observations are consistent.
   ======================================================================

Risk & Stochastic Choice
------------------------

**Risk Choice** --- safe vs. risky gambles:

.. code-block:: python

   from pyrevealed import RiskChoiceLog, RiskChoiceSummary
   import numpy as np

   safe_values = np.array([40, 45, 50, 55, 60, 65, 70, 75])
   risky_outcomes = np.array([
       [0, 100], [0, 100], [0, 100], [0, 100],
       [0, 120], [0, 120], [0, 140], [0, 140],
   ])
   risky_probs = np.array([[0.5, 0.5]] * 8)
   choices = np.array([False, False, False, True, False, True, False, True])

   log = RiskChoiceLog(safe_values, risky_outcomes, risky_probs, choices)
   print(RiskChoiceSummary.from_log(log))

**Stochastic Choice** --- probabilistic choice frequencies:

.. code-block:: python

   from pyrevealed import StochasticChoiceLog, StochasticChoiceSummary

   menus = [
       frozenset({0, 1, 2, 3}),
       frozenset({0, 1, 2}),
       frozenset({1, 2, 3}),
       frozenset({0, 1}),
   ]
   choice_frequencies = [
       {0: 10, 1: 30, 2: 40, 3: 20},
       {0: 15, 1: 35, 2: 50},
       {1: 25, 2: 45, 3: 30},
       {0: 30, 1: 70},
   ]

   log = StochasticChoiceLog(menus, choice_frequencies)
   print(StochasticChoiceSummary.from_log(log))

.. code-block:: text

   ======================================================================
                         STOCHASTIC CHOICE SUMMARY
   ======================================================================
   No. Menus: 4                       RUM Consistency: [-] FAIL
   Unique Items: 4                    Regularity: [-] FAIL
   Total Observations: 400            IIA: [-] FAIL
   Computation Time: 9.25 ms          Transitivity: SST
   ======================================================================

   Input Data:
   ----------------------------------------------------------------------
                           mean   std dev       min       max
     Menu Size            3.000     0.707     2.000     4.000
     Obs per Menu       100.000     0.000   100.000   100.000

     Mean Choice Entropy ....................................... 1.4270
   ...

Production Analysis
-------------------

Firm input/output data for profit maximization and cost minimization tests.

.. code-block:: python

   from pyrevealed import ProductionLog, ProductionSummary
   import numpy as np

   input_prices = np.array([[20, 50, 10], [22, 45, 10], [18, 55, 10], [20, 50, 8], [20, 50, 12]])
   input_quantities = np.array([[100, 40, 200], [90, 45, 200], [110, 35, 200], [100, 40, 250], [100, 40, 150]])
   output_prices = np.array([[100]] * 5)
   output_quantities = np.array([[50], [48], [49], [55], [45]])

   log = ProductionLog(input_prices, input_quantities, output_prices, output_quantities)
   print(ProductionSummary.from_log(log))

.. code-block:: text

   ======================================================================
                             PRODUCTION SUMMARY
   ======================================================================
   No. Observations: 5                Profit Max: [-] FAIL
   No. Inputs: 3                      Cost Min: [-] FAIL
   No. Outputs: 1                     Returns to Scale: Decreasing
   Computation Time: 284.49 ms        Profit Efficiency: 1.0000
   ======================================================================

   Input Data:
   ----------------------------------------------------------------------
                           mean   std dev       min       max
     Input Prices        26.667    17.126     8.000    55.000
     Output Prices      100.000     0.000   100.000   100.000
     Profit           -1002.000   276.416 -1300.000  -500.000
   ...

Dataset Loaders
---------------

Three real-world datasets are available (data must be downloaded separately):

.. list-table::
   :header-rows: 1
   :widths: 20 40 15 15 10

   * - Dataset
     - Description
     - Users
     - Goods
     - Periods
   * - ``load_dunnhumby()``
     - Grocery purchases (Kaggle)
     - 2,500
     - 10
     - 104 weeks
   * - ``load_open_ecommerce()``
     - Amazon purchases (Crowdsourced)
     - 4,700
     - 50
     - 66 months
   * - ``load_uci_retail()``
     - UK online retail (UCI)
     - 1,800
     - 50
     - 13 months

.. code-block:: python

   from pyrevealed.datasets import load_dunnhumby, list_datasets

   # See available datasets
   for ds in list_datasets():
       print(f"{ds['name']}: {ds['description']}")

   # Load with options
   panel = load_dunnhumby(
       n_households=100,    # subsample
       min_weeks=10,        # minimum data per household
       period="month",      # monthly temporal breakdown
   )
