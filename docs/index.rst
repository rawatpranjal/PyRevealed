PyRevealed
==========

Analyze user choices to detect bots, confused users, and preference patterns.

**What it does:** Given purchase history (prices + quantities), PyRevealed tests if choices are internally consistent—and quantifies how "rational" the behavior is.

**Use cases:**

- Bot detection (bots often fail consistency tests)
- UX auditing (confused users show exploitable patterns)
- Preference modeling (recover utility functions from data)

Installation
------------

.. code-block:: bash

   pip install pyrevealed

For visualization support:

.. code-block:: bash

   pip install pyrevealed[viz]

Quick Example
-------------

.. code-block:: python

   from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
   import numpy as np

   log = BehaviorLog(
       cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
       action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
   )

   is_consistent = validate_consistency(log)
   integrity = compute_integrity_score(log)

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   quickstart
   tutorial
   theory
   api
   scaling
   troubleshooting
   case_study

----

Based on `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_.
Thanks to Professor Chris Chambers at Georgetown.
