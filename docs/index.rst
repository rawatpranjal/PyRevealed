PyRevealed
==========

Rationality scores for every user, at scale. Rust engine, Python interface.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <span class="feature-icon">&#10003;</span>
       <h3>Consistency Testing</h3>
       <p>GARP, SARP, HARP axioms. Verify if choices are consistent with utility maximization.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">&#9673;</span>
       <h3>Efficiency Metrics</h3>
       <p>CCEI (0-1 consistency), MPI (exploitability), Houtman-Maks (outlier fraction).</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">&#9638;</span>
       <h3>Batch Scoring</h3>
       <p>Score millions of users in parallel. Streaming chunks, bounded memory.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon">&#9889;</span>
       <h3>Rust Engine</h3>
       <p>Graph algorithms + HiGHS LP solver in Rust. 40x faster than pure Python.</p>
     </div>
   </div>

Installation
------------

.. code-block:: bash

   pip install pyrevealed

Batch Scoring Example
---------------------

.. code-block:: python

   from pyrevealed.engine import Engine
   import numpy as np

   # Each user: (prices T x K, quantities T x K)
   users = [
       (np.random.rand(20, 5) + 0.1, np.random.rand(20, 5) + 0.1)
       for _ in range(1000)
   ]

   engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
   results = engine.analyze_arrays(users)

   for r in results[:5]:
       print(f"GARP={r.is_garp}  CCEI={r.ccei:.3f}  MPI={r.mpi:.3f}  HARP={r.is_harp}")

.. code-block:: text

   GARP=False  CCEI=0.755  MPI=0.285  HARP=False
   GARP=False  CCEI=0.799  MPI=0.241  HARP=False
   GARP=False  CCEI=0.910  MPI=0.257  HARP=False
   GARP=False  CCEI=0.785  MPI=0.318  HARP=False
   GARP=False  CCEI=0.716  MPI=0.356  HARP=False

Single-User Analysis
--------------------

.. code-block:: python

   from pyrevealed import BehaviorLog, check_garp, compute_aei, compute_mpi

   log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
   garp = check_garp(log)
   aei = compute_aei(log)
   mpi = compute_mpi(log)

   print(f"Consistent: {garp.is_consistent}")
   print(f"Efficiency: {aei.efficiency_index:.3f}")
   print(f"Exploitability: {mpi.mpi_value:.3f}")

Scores
------

.. list-table::
   :header-rows: 1

   * - Score
     - Field
     - What it measures
     - Range
   * - Consistency
     - ``is_garp``
     - Are choices rationalizable? (GARP)
     - bool
   * - Efficiency
     - ``ccei``
     - How close to perfectly rational? (Afriat)
     - 0--1
   * - Exploitability
     - ``mpi``
     - Value left on the table per choice (Karp cycle)
     - 0--1
   * - Homotheticity
     - ``is_harp``
     - Do preferences scale with budget?
     - bool
   * - Noise fraction
     - ``hm_consistent / hm_total``
     - Fraction of rationalizable choices
     - 0--1

See :doc:`theory` for the economics behind each score.

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
