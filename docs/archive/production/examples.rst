Examples
========

ProductionLog
-------------

Create a production dataset with input/output prices and quantities:

.. code-block:: python

   import numpy as np
   from prefgraph import ProductionLog

   log = ProductionLog(
       input_prices=np.array([
           [20.0, 50.0],   # Period 1: wage=$20, rental=$50
           [22.0, 48.0],   # Period 2
           [21.0, 52.0],   # Period 3
       ]),
       input_quantities=np.array([
           [100.0, 50.0],  # 100 labor, 50 capital
           [90.0, 55.0],
           [110.0, 45.0],
       ]),
       output_prices=np.array([[10.0], [11.0], [10.5]]),
       output_quantities=np.array([[500.0], [480.0], [520.0]]),
       firm_id="factory_1",
   )
   print(f"Observations: {log.num_observations}")
   print(f"Inputs: {log.num_inputs}, Outputs: {log.num_outputs}")

.. code-block:: text

   Observations: 3
   Inputs: 2, Outputs: 1

Profit Maximization Test
------------------------

.. code-block:: python

   from prefgraph import test_profit_maximization

   result = test_profit_maximization(log)
   print(f"Profit maximizing: {result.is_profit_maximizing}")
   print(f"Violations: {result.violations}")

.. code-block:: text

   Profit maximizing: True
   Violations: []

Cost Minimization
-----------------

.. code-block:: python

   from prefgraph import check_cost_minimization

   result = check_cost_minimization(log)
   print(f"Cost minimizing: {result['is_cost_minimizing']}")
   print(f"Violations: {result['num_violations']}")

.. code-block:: text

   Cost minimizing: False
   Violations: 3

Returns to Scale
----------------

.. code-block:: python

   from prefgraph import estimate_returns_to_scale

   rts = estimate_returns_to_scale(log)
   print(f"Returns to scale: {rts}")

.. code-block:: text

   Returns to scale: increasing

Technical Efficiency
--------------------

.. code-block:: python

   from prefgraph import compute_technical_efficiency

   efficiencies = compute_technical_efficiency(log, method="output_oriented")
   for t, eff in enumerate(efficiencies):
       print(f"Period {t+1}: {eff:.2%}")

.. code-block:: text

   Period 1: 100.00%
   Period 2: 100.00%
   Period 3: 100.00%

Batch Scoring (Engine)
----------------------

Score many firms in parallel:

.. code-block:: python

   import numpy as np
   from prefgraph import ProductionLog
   from prefgraph.algorithms.production import test_profit_maximization

   rng = np.random.RandomState(42)
   firms = [
       ProductionLog(
           input_prices=rng.rand(10, 3) + 0.5,
           input_quantities=rng.rand(10, 3) + 0.1,
           output_prices=rng.rand(10, 2) + 1.0,
           output_quantities=rng.rand(10, 2) + 0.1,
       )
       for _ in range(5)
   ]

   for i, log in enumerate(firms):
       result = test_profit_maximization(log)
       print(f"Firm {i}: profit_max={result.is_consistent}  violations={result.num_violations}")

.. code-block:: text

   Firm 0: profit_max=False  violations=15
   Firm 1: profit_max=False  violations=12
   Firm 2: profit_max=False  violations=18
   Firm 3: profit_max=False  violations=14
   Firm 4: profit_max=False  violations=16
