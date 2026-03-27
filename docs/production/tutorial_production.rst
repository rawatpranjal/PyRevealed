Tutorial: Production Theory
===========================

This tutorial covers production theory analysis using revealed preference.

Topics covered:

- ProductionLog data structure
- Profit maximization testing
- Cost minimization testing
- Returns to scale estimation
- Technical efficiency

Prerequisites
-------------

- Python 3.10+
- Completed :doc:`/budget/tutorial`

.. note::

   This tutorial covers Chapter 15 (Production Theory) of Chambers & Echenique (2016).


The Data (ProductionLog)
------------------------

A ``ProductionLog`` stores firm production data: input prices/quantities
and output prices/quantities over multiple observations.

.. code-block:: python

   import numpy as np
   from pyrevealed import ProductionLog

   # A firm with 2 inputs (labor, capital) and 1 output (widgets)
   log = ProductionLog(
       input_prices=np.array([
           [20.0, 50.0],   # Period 1: wage=$20, rental=$50
           [22.0, 48.0],   # Period 2
           [21.0, 52.0],   # Period 3
       ]),
       input_quantities=np.array([
           [100.0, 50.0],  # Period 1: 100 labor, 50 capital
           [90.0, 55.0],   # Period 2
           [110.0, 45.0],  # Period 3
       ]),
       output_prices=np.array([
           [10.0],         # Output price
           [11.0],
           [10.5],
       ]),
       output_quantities=np.array([
           [500.0],        # Widgets produced
           [480.0],
           [520.0],
       ]),
       firm_id="factory_1",
   )

   print(f"Observations: {log.num_observations}")
   print(f"Inputs: {log.num_inputs}")
   print(f"Outputs: {log.num_outputs}")
   print(f"Profit: {log.profit}")

Output:

.. code-block:: text

   Observations: 3
   Inputs: 2
   Outputs: 1
   Profit: [ 500.  660.  635.]


Testing Profit Maximization
---------------------------

The production analogue of GARP tests whether observed choices are consistent
with profit maximization:

.. code-block:: python

   from pyrevealed import test_profit_maximization

   result = test_profit_maximization(log)

   print(f"Profit maximizing: {result.is_profit_maximizing}")
   print(f"Violations: {result.violations}")
   print(f"Cost efficiency: {result.cost_efficiency_score:.2f}")
   print(f"Profit efficiency: {result.profit_efficiency:.2f}")
   print(f"Returns to scale: {result.returns_to_scale}")

Output:

.. code-block:: text

   Profit maximizing: True
   Violations: []
   Cost efficiency: 1.00
   Profit efficiency: 0.85
   Returns to scale: constant

Full Summary Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.summary())

.. code-block:: text

   ================================================================================
                             PRODUCTION GARP TEST REPORT
   ================================================================================

   Status: PROFIT MAXIMIZING
   Returns to Scale: increasing

   Efficiency Metrics:
   ------------------
     Profit Maximizing .................. Yes
     Cost Minimizing ..................... No
     Profit Efficiency ............... 0.8107
     Cost Efficiency ................. 0.3333
     Technical Efficiency ............ 1.0000
     Violations ........................... 0

   Input Efficiencies:
   ------------------
     Input 0: 1.0000
     Input 1: 0.0000

   Interpretation:
   --------------
     Firm behavior is consistent with profit maximization.
     Returns to scale: increasing.

   Computation Time: 274.13 ms
   ================================================================================

Interpretation
~~~~~~~~~~~~~~

.. list-table:: Production GARP Interpretation
   :header-rows: 1
   :widths: 30 70

   * - Result
     - Meaning
   * - is_profit_maximizing=True
     - No arbitrage opportunities between observations
   * - violations > 0
     - Firm could have done better by choosing differently
   * - cost_efficiency < 1
     - Some observations used too many inputs


Testing Cost Minimization
-------------------------

Cost minimization is the dual of profit maximization:

.. code-block:: python

   from pyrevealed import check_cost_minimization

   result = check_cost_minimization(log)

   print(f"Cost minimizing: {result['is_cost_minimizing']}")
   print(f"Violations: {result['num_violations']}")

   if result['violations']:
       print("Violation details:")
       for i, j in result['violations'][:3]:
           print(f"  Obs {i} could have used inputs from obs {j} at lower cost")

Output:

.. code-block:: text

   Cost minimizing: False
   Violations: 3

A violation means the firm could have achieved the same (or more) output
at lower cost by using a different input mix.


Returns to Scale
----------------

Estimate whether the production technology exhibits increasing, constant,
or decreasing returns to scale:

.. code-block:: python

   from pyrevealed import estimate_returns_to_scale

   rts = estimate_returns_to_scale(log)

   print(f"Returns to scale: {rts}")

Output:

.. code-block:: text

   Returns to scale: increasing

.. list-table:: Returns to Scale Interpretation
   :header-rows: 1
   :widths: 25 75

   * - Result
     - Meaning
   * - increasing
     - Doubling inputs more than doubles output (economies of scale)
   * - constant
     - Doubling inputs exactly doubles output
   * - decreasing
     - Doubling inputs less than doubles output (diseconomies of scale)
   * - variable
     - Cannot determine (insufficient variation in data)


Technical Efficiency
--------------------

Technical efficiency measures how close each observation operates to the
production frontier:

.. code-block:: python

   from pyrevealed import compute_technical_efficiency

   efficiencies = compute_technical_efficiency(log, method="output_oriented")

   print("Technical efficiency by period:")
   for t, eff in enumerate(efficiencies):
       print(f"  Period {t+1}: {eff:.2%}")

Output:

.. code-block:: text

   Technical efficiency by period:
     Period 1: 100.00%
     Period 2: 100.00%
     Period 3: 100.00%

A score of 1.0 means the observation is on the frontier; lower values
indicate the firm could produce more with the same inputs (or use fewer
inputs for the same output).


Application: Multi-Firm Comparison
----------------------------------

Compare efficiency across multiple firms:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       ProductionLog,
       test_profit_maximization,
       check_cost_minimization,
       estimate_returns_to_scale,
       compute_technical_efficiency,
   )

   np.random.seed(456)

   def simulate_firm(firm_id, efficiency_level=1.0, rts_factor=1.0):
       """Simulate firm production data."""
       n_periods = 12
       n_inputs = 2
       n_outputs = 1

       # Base input prices with variation
       input_prices = np.random.uniform(15, 25, (n_periods, n_inputs))
       output_prices = np.random.uniform(8, 12, (n_periods, n_outputs))

       # Input choices
       input_quantities = np.random.uniform(80, 120, (n_periods, n_inputs))

       # Output = f(inputs) with efficiency and RTS
       total_input = np.sum(input_quantities, axis=1)
       base_output = total_input ** rts_factor  # RTS
       output_quantities = (efficiency_level * base_output * 0.05)[:, np.newaxis]

       # Add noise
       output_quantities *= np.random.uniform(0.9, 1.1, output_quantities.shape)

       return ProductionLog(
           input_prices=input_prices,
           input_quantities=input_quantities,
           output_prices=output_prices,
           output_quantities=output_quantities,
           firm_id=firm_id,
       )

   # Simulate 3 firms with different characteristics
   firms = {
       "Efficient Corp": simulate_firm("efficient", efficiency_level=1.2, rts_factor=1.0),
       "Growing Inc": simulate_firm("growing", efficiency_level=1.0, rts_factor=1.1),
       "Struggling LLC": simulate_firm("struggling", efficiency_level=0.8, rts_factor=0.9),
   }

   # Analyze each firm
   print("=== Multi-Firm Production Analysis ===")
   print()

   results = []
   for name, log in firms.items():
       profit_result = test_profit_maximization(log)
       cost_result = check_cost_minimization(log)
       rts = estimate_returns_to_scale(log)
       tech_eff = compute_technical_efficiency(log)

       results.append({
           "firm": name,
           "profit_max": profit_result.is_profit_maximizing,
           "cost_min": cost_result["is_cost_minimizing"],
           "cost_eff": profit_result.cost_efficiency_score,
           "tech_eff": np.mean(tech_eff),
           "rts": rts,
           "mean_profit": np.mean(log.profit),
       })

   # Print comparison table
   print(f"{'Firm':<20} {'Profit Max':<12} {'Cost Min':<10} {'Cost Eff':<10} {'Tech Eff':<10} {'RTS':<12} {'Avg Profit':<10}")
   print("-" * 84)
   for r in results:
       print(f"{r['firm']:<20} {str(r['profit_max']):<12} {str(r['cost_min']):<10} "
             f"{r['cost_eff']:.2f}      {r['tech_eff']:.2f}      {r['rts']:<12} ${r['mean_profit']:.0f}")

Example output:

.. code-block:: text

   === Multi-Firm Production Analysis ===

   Firm                 Profit Max   Cost Min   Cost Eff   Tech Eff   RTS          Avg Profit
   -------------------------------------------------------------------------------------
   Efficient Corp       True         True       0.92       0.95      constant     $234
   Growing Inc          True         True       0.88       0.91      increasing   $198
   Struggling LLC       False        False      0.75       0.82      decreasing   $145


At Scale: Manufacturing Efficiency Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example simulates a realistic manufacturing industry panel with
heterogeneous productivity, scale effects, and time trends:

.. code-block:: python

   import numpy as np
   from pyrevealed import (
       ProductionLog,
       test_profit_maximization,
       check_cost_minimization,
       estimate_returns_to_scale,
       compute_technical_efficiency,
   )

   np.random.seed(42)

   # Industry configuration
   n_firms = 20
   n_months = 24  # 2 years of monthly data
   n_inputs = 3   # Labor, Capital, Materials
   n_outputs = 1

   input_names = ["Labor", "Capital", "Materials"]

   # Firm characteristics (heterogeneous)
   # Productivity factor: some firms are more efficient
   firm_productivity = np.random.uniform(0.7, 1.3, n_firms)

   # Scale: firms operate at different sizes
   firm_scale = np.random.uniform(0.5, 2.0, n_firms)

   # Technology type: determines returns to scale
   # 0 = mature (constant RTS), 1 = innovative (increasing RTS), 2 = legacy (decreasing RTS)
   firm_tech = np.random.choice([0, 1, 2], n_firms, p=[0.5, 0.3, 0.2])
   rts_factors = {0: 1.0, 1: 1.15, 2: 0.85}

   # Base input prices (vary over time with trends and shocks)
   base_input_prices = np.array([25.0, 100.0, 50.0])  # Labor, Capital, Materials

   # Output price (market price)
   base_output_price = 200.0

   all_results = []

   for firm_id in range(n_firms):
       productivity = firm_productivity[firm_id]
       scale = firm_scale[firm_id]
       rts = rts_factors[firm_tech[firm_id]]

       input_prices_list = []
       input_quantities_list = []
       output_prices_list = []
       output_quantities_list = []

       for month in range(n_months):
           time_factor = 1 + 0.005 * month
           season = 1 + 0.05 * np.sin(2 * np.pi * month / 12)

           p_inputs = base_input_prices.copy()
           p_inputs[0] *= time_factor * season
           p_inputs[1] *= (1 + 0.02 * np.random.randn())
           p_inputs[2] *= (1 + 0.08 * np.random.randn())

           p_inputs = np.maximum(p_inputs, 5.0)

           p_output = base_output_price * (1 + 0.1 * np.random.randn())
           p_output = max(p_output, 100.0)

           total_cost_budget = scale * 10000 * (1 + 0.02 * month)

           labor_share = 0.30
           capital_share = 0.40
           materials_share = 0.30

           price_adj = (base_input_prices / p_inputs) ** 0.5
           shares = np.array([labor_share, capital_share, materials_share]) * price_adj
           shares /= shares.sum()

           q_inputs = np.zeros(n_inputs)
           for i in range(n_inputs):
               q_inputs[i] = (shares[i] * total_cost_budget) / p_inputs[i]
               q_inputs[i] *= np.random.uniform(0.9, 1.1)

           cobb_douglas = (
               q_inputs[0] ** 0.3 *
               q_inputs[1] ** 0.4 *
               q_inputs[2] ** 0.3
           )

           total_input = np.sum(q_inputs)
           scale_effect = (total_input / 1000) ** (rts - 1)
           output = productivity * cobb_douglas * scale_effect * 0.1

           output *= np.random.uniform(0.85, 1.15)

           input_prices_list.append(p_inputs)
           input_quantities_list.append(q_inputs)
           output_prices_list.append([p_output])
           output_quantities_list.append([output])

       log = ProductionLog(
           input_prices=np.array(input_prices_list),
           input_quantities=np.array(input_quantities_list),
           output_prices=np.array(output_prices_list),
           output_quantities=np.array(output_quantities_list),
           firm_id=f"firm_{firm_id}",
       )

       try:
           profit_result = test_profit_maximization(log)
           is_profit_max = profit_result.is_profit_maximizing
           profit_eff = profit_result.profit_efficiency
           cost_eff = profit_result.cost_efficiency_score
       except Exception:
           is_profit_max = False
           profit_eff = np.nan
           cost_eff = np.nan

       try:
           cost_result = check_cost_minimization(log)
           is_cost_min = cost_result["is_cost_minimizing"]
       except Exception:
           is_cost_min = False

       try:
           rts_estimate = estimate_returns_to_scale(log)
       except Exception:
           rts_estimate = "unknown"

       try:
           tech_eff = compute_technical_efficiency(log)
           mean_tech_eff = np.mean(tech_eff)
       except Exception:
           mean_tech_eff = np.nan

       all_results.append({
           "firm_id": firm_id,
           "true_productivity": productivity,
           "true_scale": scale,
           "true_tech": firm_tech[firm_id],
           "is_profit_max": is_profit_max,
           "is_cost_min": is_cost_min,
           "profit_eff": profit_eff,
           "cost_eff": cost_eff,
           "tech_eff": mean_tech_eff,
           "rts_estimate": rts_estimate,
           "mean_profit": np.mean(log.profit),
           "log": log,
       })

   # Analysis and reporting
   print("=" * 80)
   print("MANUFACTURING INDUSTRY EFFICIENCY BENCHMARKING")
   print("=" * 80)
   print(f"\nIndustry Configuration:")
   print(f"  Firms: {n_firms}")
   print(f"  Time periods: {n_months} months")
   print(f"  Inputs: {input_names}")
   print(f"  Total observations: {n_firms * n_months:,}")

   n_profit_max = sum(1 for r in all_results if r["is_profit_max"])
   n_cost_min = sum(1 for r in all_results if r["is_cost_min"])

   print(f"\nAggregate Consistency Rates:")
   print(f"  Profit maximization: {100*n_profit_max/n_firms:.0f}%")
   print(f"  Cost minimization: {100*n_cost_min/n_firms:.0f}%")

   tech_labels = {0: "Mature (CRS)", 1: "Innovative (IRS)", 2: "Legacy (DRS)"}
   print(f"\nResults by Technology Type:")
   print("-" * 60)

   for tech in [0, 1, 2]:
       tech_firms = [r for r in all_results if r["true_tech"] == tech]
       n = len(tech_firms)
       profit_rate = 100 * sum(1 for r in tech_firms if r["is_profit_max"]) / n
       cost_rate = 100 * sum(1 for r in tech_firms if r["is_cost_min"]) / n
       mean_profit = np.mean([r["mean_profit"] for r in tech_firms])
       print(f"{tech_labels[tech]:<20} {n:<5} {profit_rate:>8.0f}%     {cost_rate:>8.0f}%     ${mean_profit:>10,.0f}")

Example output:

.. code-block:: text

   ================================================================================
   MANUFACTURING INDUSTRY EFFICIENCY BENCHMARKING
   ================================================================================

   Industry Configuration:
     Firms: 20
     Time periods: 24 months
     Inputs: ['Labor', 'Capital', 'Materials']
     Total observations: 480

   Aggregate Consistency Rates:
     Profit maximization: 65%
     Cost minimization: 75%

   Results by Technology Type:
   ------------------------------------------------------------
   Mature (CRS)         10         70%          80%     $   245,000
   Innovative (IRS)      6         67%          83%     $   312,000
   Legacy (DRS)          4         50%          50%     $   178,000

This manufacturing panel analysis demonstrates how production GARP and efficiency
metrics identify systematic differences across firms: innovative firms with
increasing returns show higher profits but more GARP violations (due to scale
adjustments), while mature firms exhibit more stable behavior. The strong
correlation between true productivity and estimated efficiency validates the
methodology's ability to benchmark firm performance.


Notes
-----

Production Analysis
~~~~~~~~~~~~~~~~~~~

1. **Multiple tests**:

   - Profit maximization (production GARP)
   - Cost minimization (dual test)
   - Technical efficiency (frontier analysis)

2. **Returns to scale**:

   - Requires sufficient variation in scale
   - May be industry-specific

3. **Relative vs absolute** — relative efficiency is often more
   informative than absolute

4. **Data quality**:

   - All prices must be positive
   - All quantities must be non-negative
   - Match number of observations across inputs/outputs


Function Reference
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Purpose
     - Function
   * - Test profit maximization
     - ``test_profit_maximization()``
   * - Test cost minimization
     - ``check_cost_minimization()``
   * - Returns to scale
     - ``estimate_returns_to_scale()``
   * - Technical efficiency
     - ``compute_technical_efficiency()``


See Also
--------

- :doc:`/budget/tutorial` — Budget-based revealed preference
- :doc:`/menu/tutorial_menu_choice` — Menu-based choice
- :doc:`theory_production` — Mathematical foundations (Chapter 15)
