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
       <p>Budgets, menus, stochastic choice, and production data with separability testing.</p>
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

Quick Example
-------------

.. code-block:: python

   from pyrevealed import BehaviorLog, compute_integrity_score
   import numpy as np

   log = BehaviorLog(
       cost_vectors=np.array([[1.0, 2.0], [2.0, 1.0]]),
       action_vectors=np.array([[3.0, 1.0], [1.0, 3.0]])
   )

   result = compute_integrity_score(log)
   print(result.summary())

Output:

.. code-block:: text

   ================================================================================
                            AFRIAT EFFICIENCY INDEX REPORT
   ================================================================================

   Status: PERFECT (AEI = 1.0)

   Metrics:
   -------
     Efficiency Index (AEI) .......... 1.0000
     Waste Fraction .................. 0.0000
     Perfectly Consistent ............... Yes
     Binary Search Iterations ............. 0
     Tolerance ................... 1.0000e-06

   Interpretation:
   --------------
     Perfect consistency - behavior fully rationalized by utility maximization

   Computation Time: 0.30 ms
   ================================================================================

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
   **Quick interpretation**: Integrity ≥0.95 is excellent, ≥0.90 is good, <0.70 indicates problems.
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

----

Based on `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_.
Validated against R's `revealedPrefs <https://cran.r-project.org/package=revealedPrefs>`_ package. See :doc:`validation`.
