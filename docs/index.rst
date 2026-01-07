PyRevealed
==========

Test if user choices are internally consistent and quantify how "rational" the behavior is.

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
