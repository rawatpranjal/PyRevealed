PyRevealed
==========

Revealed preference analysis for panel data. Test behavioral consistency across thousands of users and time periods.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <span class="feature-icon">&#10003;</span>
       <h3>Consistency Testing</h3>
       <p>GARP, WARP, SARP axioms. Verify if choices are consistent with utility maximization.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">&#9673;</span>
       <h3>Efficiency Metrics</h3>
       <p>AEI (0-1 consistency), MPI (welfare loss), Houtman-Maks (outlier fraction).</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">&#9638;</span>
       <h3>Panel Analysis</h3>
       <p>Users x periods x choices. Parallelized engine with aggregate and individual reports.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">&#9889;</span>
       <h3>Production Ready</h3>
       <p>Numba-accelerated. Handles 5,000+ observations per user, thousands of users.</p>
     </div>
   </div>

Installation
------------

.. code-block:: bash

   pip install pyrevealed
   pip install pyrevealed[datasets]   # for built-in panel datasets

Panel Analysis Example
----------------------

.. code-block:: python

   from pyrevealed.datasets import load_dunnhumby

   panel = load_dunnhumby(n_households=100)   # 85 households, 10 goods, ~50 weeks each
   report = panel.summary()
   print(report)

.. code-block:: text

   ======================================================================
                               PANEL SUMMARY
   ======================================================================
   No. Users: 85                      GARP Pass Rate: 4.7%
   Total Observations: 4,174          Mean AEI: 0.8237
   No. Goods: 10                      Mean MPI: 0.2412
   Obs/User (mean): 49.1              Computation Time: 4.27 s
   ======================================================================

   Consistency Rates:
   ----------------------------------------------------------------------
     GARP ............................................... 4.7% (4 / 85)
     WARP ............................................... 5.9% (5 / 85)
     SARP ............................................... 1.2% (1 / 85)

   Efficiency Distribution:
   ----------------------------------------------------------------------
                       mean     std     min     25%     50%     75%     max
     AEI              0.824   0.104   0.598   0.743   0.820   0.906   1.000
     MPI              0.241   0.125   0.000   0.156   0.235   0.322   0.480
     HM Index         0.765   0.118   0.444   0.711   0.766   0.833   1.000

   Most Inconsistent (Bottom 5):
   ----------------------------------------------------------------------
       1. household_99 ..................... AEI=0.598, MPI=0.412, T=78
       2. household_17 ..................... AEI=0.624, MPI=0.432, T=64
       3. household_43 ..................... AEI=0.636, MPI=0.423, T=37
       4. household_47 ..................... AEI=0.638, MPI=0.372, T=47
       5. household_77 ..................... AEI=0.638, MPI=0.466, T=62
   ======================================================================

Supported Data Types
--------------------

- **BehaviorLog** --- Budget-based purchases (prices + quantities)
- **MenuChoiceLog** --- Discrete choices from menus (surveys, voting)
- **RiskChoiceLog** --- Safe vs. risky gambles (insurance, lotteries)
- **StochasticChoiceLog** --- Probabilistic choice frequencies
- **ProductionLog** --- Firm inputs and outputs (profit/cost tests)
- **BehaviorPanel** --- Multi-user panel container for any of the above

See :doc:`examples` for working code for each type.

----

Based on `Chambers & Echenique (2016) <https://www.amazon.com/Revealed-Preference-Econometric-Society-Monographs/dp/1107087805>`_.
Validated against R's `revealedPrefs <https://cran.r-project.org/package=revealedPrefs>`_. See :doc:`performance`.

.. toctree::
   :maxdepth: 1
   :hidden:

   quickstart
   examples
   theory
   performance
   api
   references
