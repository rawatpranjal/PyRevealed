Examples
========

Exponential Discounting
-----------------------

Test whether time-dated choices are consistent with a fixed discount factor:

.. code-block:: python

   import numpy as np
   from prefgraph.algorithms.intertemporal import test_exponential_discounting, DatedChoice

   choices = [
       DatedChoice(amounts=np.array([100.0, 105.0]), dates=np.array([0, 30]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 110.0]), dates=np.array([0, 60]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 103.0]), dates=np.array([30, 60]), chosen=1),
   ]
   result = test_exponential_discounting(choices)
   print(f"Exponential discounting: {result.is_consistent}")
   print(f"Delta range: [{result.delta_lower:.3f}, {result.delta_upper:.3f}]")

.. code-block:: text

   Exponential discounting: True
   Delta range: [0.833, 0.997]

Present Bias Detection
----------------------

Impatient now but patient later - the hallmark of present bias:

.. code-block:: python

   from prefgraph.algorithms.intertemporal import test_present_bias, DatedChoice
   import numpy as np

   choices = [
       DatedChoice(amounts=np.array([100.0, 110.0]), dates=np.array([0, 1]), chosen=0),
       DatedChoice(amounts=np.array([100.0, 110.0]), dates=np.array([30, 31]), chosen=1),
       DatedChoice(amounts=np.array([50.0, 55.0]), dates=np.array([0, 7]), chosen=0),
       DatedChoice(amounts=np.array([50.0, 55.0]), dates=np.array([60, 67]), chosen=1),
   ]
   result = test_present_bias(choices, threshold=0.1)
   print(f"Present bias: {result['has_present_bias']}")
   print(f"Immediate patience: {result['immediate_patience']:.0%}")
   print(f"Future patience: {result['future_patience']:.0%}")

.. code-block:: text

   Present bias: True
   Immediate patience: 0%
   Future patience: 100%

Quasi-Hyperbolic (Beta-Delta) Model
------------------------------------

Test for the Laibson (1997) beta-delta model:

.. code-block:: python

   from prefgraph.algorithms.intertemporal import test_quasi_hyperbolic, DatedChoice
   import numpy as np

   choices = [
       DatedChoice(amounts=np.array([100.0, 115.0]), dates=np.array([0, 30]), chosen=0),
       DatedChoice(amounts=np.array([100.0, 108.0]), dates=np.array([0, 14]), chosen=0),
       DatedChoice(amounts=np.array([100.0, 105.0]), dates=np.array([60, 90]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 103.0]), dates=np.array([90, 120]), chosen=1),
   ]
   result = test_quasi_hyperbolic(choices)
   print(f"Beta-delta consistent: {result.is_consistent}")
   print(f"Beta: [{result.beta_lower:.3f}, {result.beta_upper:.3f}]")
   print(f"Delta: [{result.delta_lower:.3f}, {result.delta_upper:.3f}]")

.. code-block:: text

   Beta-delta consistent: True
   Beta: [0.010, 0.870]
   Delta: [0.000, 1.000]

Recovering Discount Factors
----------------------------

Bound the discount factor from observed choices:

.. code-block:: python

   from prefgraph.algorithms.intertemporal import recover_discount_factor, DatedChoice
   import numpy as np

   choices = [
       DatedChoice(amounts=np.array([100.0, 102.0]), dates=np.array([0, 30]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 105.0]), dates=np.array([0, 60]), chosen=1),
       DatedChoice(amounts=np.array([100.0, 101.0]), dates=np.array([0, 30]), chosen=0),
   ]
   bounds = recover_discount_factor(choices)
   print(f"Delta: [{bounds.delta_lower:.4f}, {bounds.delta_upper:.4f}]")
   print(f"Midpoint: {bounds.midpoint:.4f}")

.. code-block:: text

   Delta: [0.9804, 0.9901]
   Midpoint: 0.9852
