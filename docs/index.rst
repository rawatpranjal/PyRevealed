PyRevealed
==========

.. raw:: html

   <span class="speed-badge">132x faster than R</span>

.. raw:: html

   <p class="hero-tagline">
   Production-ready revealed preference analysis. Test if choices are internally consistent,
   quantify behavioral rationality, and detect exploitable patterns in decision data.
   </p>

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>Consistency Testing</h3>
       <p>Check GARP, WARP, and SARP axioms to verify if choices could come from utility maximization.</p>
     </div>
     <div class="feature-card">
       <h3>Integrity Scoring</h3>
       <p>Get a 0-1 score measuring how close behavior is to perfect rationality (Afriat Efficiency Index).</p>
     </div>
     <div class="feature-card">
       <h3>Exploitability Analysis</h3>
       <p>Compute the Money Pump Index to find how much value can be extracted from inconsistent choices.</p>
     </div>
     <div class="feature-card">
       <h3>Utility Recovery</h3>
       <p>Reconstruct utility functions that rationalize observed behavior for prediction and simulation.</p>
     </div>
   </div>

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

   # Two purchase observations: prices and quantities
   log = BehaviorLog(
       cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
       action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
   )

   # Check if choices are consistent with utility maximization
   is_consistent = validate_consistency(log)  # True

   # Get integrity score (0 = irrational, 1 = perfectly rational)
   result = compute_integrity_score(log)
   print(f"Integrity: {result.efficiency_index:.2f}")  # 1.00

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   quickstart
   tutorial
   tutorial_ecommerce
   theory
   api
   scaling
   validation
   troubleshooting
   case_study

----

Based on `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_.
Validated against R's `revealedPrefs <https://cran.r-project.org/package=revealedPrefs>`_ package. See :doc:`validation`.
