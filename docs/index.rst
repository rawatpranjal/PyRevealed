PyRevealed
==========

Behavioral Signal Analysis for User Understanding.

Given a history of user choices and the options available at each choice, PyRevealed computes:

- **Consistency scores**: How internally consistent is this user's behavior?
- **Preference recovery**: What utility function explains their choices?
- **Exploitability metrics**: How much could be extracted via arbitrage?
- **Feature independence**: Are choices over group A independent of group B?

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

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   tutorial
   theory
   api
   case_study

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
