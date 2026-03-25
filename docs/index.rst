PyRevealed
==========

.. raw:: html

   <span class="speed-badge">Faster than R</span>

.. raw:: html

   <p class="hero-tagline">
   Production-ready revealed preference analysis. Test if choices are internally consistent,
   quantify behavioral consistency, and analyze decision patterns.
   </p>

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <span class="feature-icon">✓</span>
       <h3>Consistency Testing</h3>
       <p>Check GARP, WARP, and SARP axioms to verify if choices could come from utility maximization.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">◉</span>
       <h3>Behavioral Metrics</h3>
       <p>Get AEI (0-1 consistency score) and MPI (welfare loss from preference cycles).</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">ƒ</span>
       <h3>Utility Recovery</h3>
       <p>Reconstruct utility functions that rationalize observed behavior for prediction and simulation.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">⚙</span>
       <h3>ML Integration</h3>
       <p>sklearn-compatible PreferenceEncoder for extracting behavioral features into ML pipelines.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">☰</span>
       <h3>Multiple Data Types</h3>
       <p>Budgets, menus, stochastic choice, risk choice, and production data.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">⚡</span>
       <h3>Production Ready</h3>
       <p>Fast parallel processing for thousands of users. Cross-validated against R's revealedPrefs.</p>
     </div>
   </div>

Installation
------------

.. code-block:: bash

   pip install pyrevealed

For visualization support:

.. code-block:: bash

   pip install pyrevealed[viz]

All Data Types
--------------

PyRevealed supports 5 data types. Each provides a comprehensive ``.summary()`` report.

**1. Budget-Based Behavior (BehaviorLog)**

For consumer purchases with prices and quantities. This is the standard revealed preference setting where budget constraints determine what is affordable.

.. code-block:: python

   from pyrevealed import BehaviorLog
   import numpy as np

   # Coffee shop visits: 6 trips, 4 items (latte, cappuccino, pastry, sandwich)
   prices = np.array([
       [5.00, 4.50, 3.00, 8.00],  # Regular prices
       [4.00, 4.50, 3.00, 8.00],  # Latte discount
       [5.00, 3.50, 3.00, 8.00],  # Cappuccino discount
       [5.00, 4.50, 2.00, 8.00],  # Pastry discount
       [5.00, 4.50, 3.00, 6.00],  # Sandwich discount
       [4.50, 4.00, 2.50, 7.00],  # Everything 10% off
   ])
   quantities = np.array([
       [1, 1, 1, 0],  # Two drinks + pastry
       [2, 0, 1, 0],  # More lattes when cheap
       [0, 2, 1, 0],  # More cappuccinos when cheap
       [1, 1, 2, 0],  # More pastries when cheap
       [1, 0, 1, 1],  # Add sandwich when cheap
       [1, 1, 2, 0],  # General savings -> more pastries
   ])

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
   print(log.summary())

.. code-block:: text

   ======================================================================
                             BEHAVIORAL SUMMARY
   ======================================================================
   User ID: N/A                       GARP: [+] PASS
   No. Observations: 6                WARP: [+] PASS
   No. Goods: 4                       SARP: [-] FAIL
   Method: Floyd-Warshall             AEI: 1.0000
   Computation Time: 234.17 ms        MPI: 0.0000
   ======================================================================

   Input Data:
   ----------------------------------------------------------------------
                           mean   std dev       min       max
     Prices               4.812     1.790     2.000     8.000
     Quantities           0.833     0.687     0.000     2.000
     Expenditure         12.417     1.455    10.000    14.000

   Revealed Preference Graph:
   ----------------------------------------------------------------------
     R  (direct, p'x >= p'y) .................... 18 / 36 edges (50.0%)
     P  (strict, p'x >  p'y) .................... 10 / 36 edges (27.8%)
     R* (transitive closure) .................... 18 / 36 edges (50.0%)
     Violation pairs (R* & P') ...................................... 0

   Consistency Tests:
   ----------------------------------------------------------------------
     GARP .................................................... [+] PASS
     WARP .................................................... [+] PASS
     SARP .......................................... [-] FAIL (1 cycle)

   Goodness-of-Fit:
   ----------------------------------------------------------------------
     Afriat Efficiency (AEI) ................................... 1.0000
       Binary search iterations ..................................... 0
       Budget waste ............................................. 0.00%
     Money Pump Index (MPI) .................................... 0.0000
       Violation cycles ............................................. 0
       Total expenditure ....................................... $74.50

   Interpretation:
   ----------------------------------------------------------------------
     Perfect consistency - behavior fully rationalized by utility maximization
   ======================================================================

**2. Menu-Based Choice (MenuChoiceLog)**

For discrete choices from menus without prices. Common in experiments and surveys where participants choose from option sets.

.. code-block:: python

   from pyrevealed import MenuChoiceLog

   # Restaurant menu choices: 6 visits, 4 dishes (pasta, steak, salad, fish)
   menus = [
       frozenset({0, 1, 2, 3}),  # Full menu
       frozenset({0, 1, 2}),     # No fish today
       frozenset({0, 2, 3}),     # No steak today
       frozenset({1, 2, 3}),     # No pasta today
       frozenset({0, 1}),        # Limited: pasta or steak
       frozenset({2, 3}),        # Limited: salad or fish
   ]
   choices = [1, 1, 0, 1, 1, 3]  # Prefers steak > pasta > fish > salad

   log = MenuChoiceLog(menus, choices)
   print(log.summary())

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

**3. Risk Choice (RiskChoiceLog)**

For choices between safe and risky options. Reveals risk attitudes and tests Expected Utility axioms.

.. code-block:: python

   from pyrevealed import RiskChoiceLog
   import numpy as np

   # Insurance decisions: 8 scenarios with varying premiums and coverage
   safe_values = np.array([40, 45, 50, 55, 60, 65, 70, 75])  # Certain payoff
   risky_outcomes = np.array([
       [0, 100], [0, 100], [0, 100], [0, 100],
       [0, 120], [0, 120], [0, 140], [0, 140],
   ])  # Risky lottery outcomes
   risky_probs = np.array([[0.5, 0.5]] * 8)  # 50-50 chance

   # Risk-averse: prefers safe when expected values are close
   choices = np.array([False, False, False, True, False, True, False, True])

   log = RiskChoiceLog(safe_values, risky_outcomes, risky_probs, choices)
   print(log.summary())

.. code-block:: text

   ======================================================================
                            RISK CHOICE SUMMARY
   ======================================================================
   No. Observations: 8                Risk Category: Risk Averse
   Risk-Seeking Choices: 3            Risk Aversion (rho): 0.6941
   Risk-Averse Choices: 2             Consistency: 0.6250
   Computation Time: 0.33 ms          EU Axioms: [+] PASS
   ======================================================================

   Input Data:
   ----------------------------------------------------------------------
                           mean   std dev       min       max
     Safe Values         57.500    11.456    40.000    75.000
     Risky EV            57.500     8.292    50.000    70.000

   Choice Distribution:
   ----------------------------------------------------------------------
     Risk-Seeking ........................................... 3 (37.5%)
     Risk-Averse ............................................ 2 (25.0%)
     Risk-Neutral ........................................... 3 (37.5%)

   Risk Profile (CRRA):
   ----------------------------------------------------------------------
     Risk Category ........................................ Risk Averse
     Risk Aversion (rho) ....................................... 0.6941
     Consistency Score ......................................... 0.6250

   Expected Utility Axioms:
   ----------------------------------------------------------------------
     Status ............................................. [+] SATISFIED

   Interpretation:
   ----------------------------------------------------------------------
     Decision-maker prefers certainty over gambles.
     Certainty premium: ~63% less for certainty.
     Model fit: 62.5% of choices consistent with CRRA profile.
   ======================================================================

**4. Stochastic Choice (StochasticChoiceLog)**

For probabilistic choices with observed frequencies. Tests Random Utility Model consistency.

.. code-block:: python

   from pyrevealed import StochasticChoiceLog

   # Product catalog: 4 menus, ~100 observations each
   # Items: 0=Basic, 1=Standard, 2=Premium, 3=Deluxe
   menus = [
       frozenset({0, 1, 2, 3}),  # Full catalog
       frozenset({0, 1, 2}),     # No deluxe
       frozenset({1, 2, 3}),     # No basic
       frozenset({0, 1}),        # Budget options only
   ]
   choice_frequencies = [
       {0: 10, 1: 30, 2: 40, 3: 20},  # Premium most popular
       {0: 15, 1: 35, 2: 50},          # Premium still top
       {1: 25, 2: 45, 3: 30},          # Premium still top
       {0: 30, 1: 70},                 # Standard beats basic
   ]

   log = StochasticChoiceLog(menus, choice_frequencies)
   print(log.summary())

.. code-block:: text

   ======================================================================
                         STOCHASTIC CHOICE SUMMARY
   ======================================================================
   No. Menus: 4                       RUM Consistency: [-] FAIL
   Unique Items: 4                    Regularity: [-] FAIL
   Total Observations: 400            IIA: [-] FAIL
   Computation Time: 4.81 ms          Transitivity: SST
   ======================================================================

   Consistency Tests:
   ----------------------------------------------------------------------
     RUM Consistency ......................................... [-] FAIL
       Distance to nearest RUM ................................. 0.1000
     Regularity (Luce) ....................................... [-] FAIL
       Regularity violations ........................................ 1
     IIA ..................................................... [-] FAIL

   Stochastic Transitivity:
   ----------------------------------------------------------------------
     Weak (WST) .............................................. [+] PASS
     Moderate (MST) .......................................... [+] PASS
     Strong (SST) ............................................ [+] PASS

   Interpretation:
   ----------------------------------------------------------------------
     Choices cannot be explained by any random utility model.
     Distance to nearest RUM: 0.1000
   ======================================================================

**5. Production/Firm (ProductionLog)**

For firm behavior with inputs and outputs. Tests profit maximization and cost minimization.

.. code-block:: python

   from pyrevealed import ProductionLog
   import numpy as np

   # Manufacturing plant: 5 periods, 3 inputs (labor, capital, materials)
   input_prices = np.array([
       [20, 50, 10],  # Base input prices
       [22, 45, 10],  # Labor up, capital down
       [18, 55, 10],  # Labor down, capital up
       [20, 50, 8],   # Materials cheaper
       [20, 50, 12],  # Materials expensive
   ])
   input_quantities = np.array([
       [100, 40, 200],  # Base production
       [90, 45, 200],   # Substitute away from labor
       [110, 35, 200],  # Substitute away from capital
       [100, 40, 250],  # Use more materials
       [100, 40, 150],  # Use less materials
   ])
   output_prices = np.array([[100], [100], [100], [100], [100]])
   output_quantities = np.array([[50], [48], [49], [55], [45]])

   log = ProductionLog(input_prices, input_quantities, output_prices, output_quantities)
   print(log.summary())

.. code-block:: text

   ======================================================================
                             PRODUCTION SUMMARY
   ======================================================================
   No. Observations: 5                Profit Max: [-] FAIL
   No. Inputs: 3                      Cost Min: [-] FAIL
   No. Outputs: 1                     Returns to Scale: Decreasing
   Computation Time: 113.64 ms        Profit Efficiency: 1.0000
   ======================================================================

   Consistency Tests:
   ----------------------------------------------------------------------
     Profit Maximization ...................... [-] FAIL (3 violations)
     Cost Minimization ....................................... [-] FAIL
     Returns to Scale ...................................... Decreasing

   Efficiency Metrics:
   ----------------------------------------------------------------------
     Technical Efficiency ...................................... 0.9960
     Cost Efficiency ........................................... 0.8000
     Profit Efficiency ......................................... 1.0000

   Per-Input Efficiency:
   ----------------------------------------------------------------------
     Input 0 ................................................... 0.0969
     Input 1 ................................................... 0.0000
     Input 2 ................................................... 0.9695

   Interpretation:
   ----------------------------------------------------------------------
     Found 3 profit maximization violation(s).
     Returns to scale: decreasing.
     Operating at 100.0% of optimal profit efficiency.
   ======================================================================

Power Analysis (Optional)
-------------------------

For rigorous analysis, include power analysis to assess how meaningful your test results are:

.. code-block:: python

   from pyrevealed import BehaviorLog
   import numpy as np

   # Same grocery shopping data
   prices = np.array([
       [3.50, 2.00, 4.00, 6.00],
       [2.80, 2.50, 4.00, 5.00],
       [3.50, 1.50, 3.50, 6.50],
       [3.00, 2.00, 5.00, 4.50],
       [4.00, 2.50, 3.00, 6.00],
   ])
   quantities = np.array([
       [2, 3, 1, 1],
       [4, 2, 1, 1],
       [2, 5, 1, 1],
       [2, 3, 1, 2],
       [2, 2, 3, 1],
   ])

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
   print(log.summary(include_power=True))

The power analysis section shows:

- **Bronars Power**: Probability that random behavior would fail GARP (higher = more demanding test)
- **Optimal Efficiency (e*)**: Efficiency level that maximizes predictive success
- **Optimal Measure (m*)**: Maximum predictive success value

Multi-User Panel Analysis
-------------------------

Use ``BehaviorPanel`` to analyze many users at once and get aggregate statistics:

.. code-block:: python

   from pyrevealed import BehaviorLog, BehaviorPanel
   import numpy as np
   np.random.seed(0)

   # Simulated household grocery data
   logs = []
   for i in range(50):
       T = np.random.randint(10, 30)
       prices = np.random.uniform(1, 10, size=(T, 5))
       quantities = np.random.uniform(0.5, 5, size=(T, 5))
       logs.append(BehaviorLog(prices, quantities, user_id=f'household_{i}'))

   panel = BehaviorPanel.from_logs(logs)
   print(panel.summary())

.. code-block:: text

   ======================================================================
                               PANEL SUMMARY
   ======================================================================
   No. Users: 50                      GARP Pass Rate: 0.0%
   Total Observations: 952            Mean AEI: 0.8298
   No. Goods: 5                       Mean MPI: 0.2402
   Obs/User (mean): 19.0              Computation Time: 609.09 ms
   ======================================================================

   Consistency Rates:
   ----------------------------------------------------------------------
     GARP ............................................... 0.0% (0 / 50)
     WARP ............................................... 0.0% (0 / 50)
     SARP ............................................... 0.0% (0 / 50)

   Efficiency Distribution:
   ----------------------------------------------------------------------
                       mean     std     min     25%     50%     75%     max
     AEI              0.830   0.076   0.620   0.776   0.832   0.866   0.990
     MPI              0.240   0.085   0.036   0.201   0.239   0.295   0.428
     HM Index         0.691   0.111   0.526   0.616   0.683   0.762   0.933

   Most Inconsistent (Bottom 5):
   ----------------------------------------------------------------------
       1. household_41 ..................... AEI=0.620, MPI=0.427, T=20
       2. household_14 ..................... AEI=0.660, MPI=0.346, T=25
       3. household_4 ...................... AEI=0.710, MPI=0.428, T=27
       4. household_3 ...................... AEI=0.729, MPI=0.279, T=19
       5. household_37 ..................... AEI=0.733, MPI=0.374, T=14
   ======================================================================

Dataset Loaders
---------------

Load real-world datasets directly into ``BehaviorPanel`` objects (requires ``pip install pyrevealed[datasets]``):

.. code-block:: python

   from pyrevealed.datasets import load_dunnhumby, list_datasets

   # See available datasets
   for ds in list_datasets():
       print(f"{ds['name']}: {ds['description']}")

   # Load Dunnhumby grocery data (2,500 households)
   panel = load_dunnhumby(n_households=100)
   print(panel.summary())

Available datasets:

- **dunnhumby**: 2,500 households, 10 grocery categories, 104 weeks (Kaggle)
- **open_ecommerce**: 4,700 consumers, 50 Amazon categories, 66 months
- **uci_retail**: 1,800 customers, 50 product categories, 13 months (UCI)

Visualizations
--------------

PyRevealed includes built-in visualizations for analysis and reporting.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;">

.. image:: _static/front_budget_sets.png
   :alt: Budget Sets - visualize budget lines and chosen bundles
   :width: 100%

.. image:: _static/front_power_analysis.png
   :alt: Power Analysis - compare against random behavior
   :width: 100%

.. image:: _static/front_aei_distribution.png
   :alt: AEI Distribution - population consistency scores
   :width: 100%

.. image:: _static/front_ccei_sensitivity.png
   :alt: CCEI Sensitivity - effect of removing outliers
   :width: 100%

.. raw:: html

   </div>

Core Functions
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Function
     - Returns
     - Score Meaning
   * - ``validate_consistency(log)``
     - ``bool``
     - True = rational
   * - ``compute_integrity_score(log)``
     - ``AEIResult`` (0-1)
     - 1 = perfect
   * - ``compute_confusion_metric(log)``
     - ``MPIResult`` (0-1)
     - 0 = no cycles
   * - ``fit_latent_values(log)``
     - ``UtilityRecoveryResult``
     - Utility values
   * - ``compute_minimal_outlier_fraction(log)``
     - ``HoutmanMaksResult`` (0-1)
     - 0 = all consistent

.. note::
   **Quick interpretation**: Integrity >=0.95 is excellent, >=0.90 is good, <0.70 indicates problems.
   Confusion <0.05 is very low, >0.15 indicates significant preference cycles.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   quickstart
   tutorials
   theory
   api
   scaling
   validation
   troubleshooting
   case_study
   references

----

Based on `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_.
Validated against R's `revealedPrefs <https://cran.r-project.org/package=revealedPrefs>`_ package. See :doc:`validation`.
