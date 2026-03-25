Tutorial 7: Uber Eats at Scale
==============================

This tutorial analyzes simulated Uber Eats order data using revealed preference
methods, with multi-core parallel processing for production-scale cohorts.

Topics covered:

- Why food-delivery data is interesting for revealed preference
- The "dense R graph" phenomenon with sparse menu choices
- Heterogeneous user simulation (churned to power users)
- Full pipeline: GARP + AEI + Houtman-Maks
- Parallel cohort analysis with ``concurrent.futures``
- Production scaling projections

Prerequisites
-------------

- Completed :doc:`tutorial` (budget-based basics)
- Python 3.10+ with NumPy
- Basic understanding of GARP and AEI

.. note::

   The full code for this tutorial is available at
   ``examples/07_uber_eats_scale.py`` in the PyRevealed repository.


Part 1: The Data Structure
--------------------------

A revealed preference "observation" at Uber Eats is one order. The user
chose item X at price P when items Y, Z were also available at their prices.

**Key structural properties:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Property
     - Description
   * - Sparse quantities
     - Users pick 1-2 items from menus of 50-200 items. Only 1-3% of the
       quantity matrix is nonzero.
   * - Price variation
     - Promotions, surge pricing, and dynamic pricing create 10-30% price
       variation across orders, which is essential for GARP violations.
   * - Dense R graph
     - Despite sparse choices, the revealed preference graph R has ~50%
       density. If you spent $25 on a burger, every item under $25 is
       "revealed affordable."
   * - High violation rates
     - Simulated users are only ~25% GARP-consistent, with mean AEI around
       0.87. Noisy discrete choices from menus generate apparent inconsistency
       even from utility-maximizing users.


Part 2: User Heterogeneity
--------------------------

Real platforms have enormous variation in user activity. The simulator draws
users from five archetypes:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Archetype
     - Share
     - Orders
     - Noise
     - Typical AEI
   * - Churned
     - 15%
     - 5-15
     - High
     - 0.95-1.0
   * - Casual
     - 30%
     - 15-60
     - Medium-high
     - 0.85-1.0
   * - Regular
     - 30%
     - 60-200
     - Medium
     - 0.75-0.90
   * - Power user
     - 15%
     - 200-500
     - Lower
     - 0.70-0.85
   * - Super-power
     - 10%
     - 500-1000
     - Low
     - 0.65-0.80

Churned users often appear GARP-consistent simply because few observations
means fewer opportunities for violations. Power users show lower AEI despite
more consistent underlying preferences, because more observations provide
more chances to detect violations.


Part 3: Running the Pipeline
-----------------------------

Single-user analysis runs GARP, AEI, and Houtman-Maks sequentially:

.. code-block:: python

   from pyrevealed import BehaviorLog, check_garp, compute_aei
   from pyrevealed.algorithms.mpi import compute_houtman_maks_index

   garp = check_garp(log)
   aei = compute_aei(log, tolerance=1e-4)
   hm = compute_houtman_maks_index(log)

For cohort analysis, each user is independent --- perfect for parallelization:

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor, as_completed

   with ProcessPoolExecutor(max_workers=8) as executor:
       futures = {executor.submit(analyze_user, log): log for log in user_logs}
       for future in as_completed(futures):
           result = future.result()

This uses ``ProcessPoolExecutor`` from the standard library (no new
dependencies). Each worker process gets its own Python interpreter,
avoiding GIL contention with Numba JIT compilation.


Part 4: Interpreting Results
----------------------------

**AEI Distribution:** A mean AEI of ~0.87 means users "waste" about 13% of
their budget from a rationality perspective. This is typical for noisy
discrete choice from menus.

**GARP Consistency:** Only ~25% of users are fully GARP-consistent. Users
with few orders (< 20) are more likely to appear consistent due to fewer
opportunities for violations.

**Houtman-Maks:** The HM fraction indicates how many orders must be removed
for consistency. High values (> 0.5) for active users suggest that menu-based
food ordering inherently generates apparent inconsistency.

**Practical signals:**

- **AEI < 0.5:** Possible bot or shared account
- **AEI 0.5-0.7:** Highly inconsistent, may indicate confused UX
- **AEI 0.7-0.9:** Normal range for active food delivery users
- **AEI > 0.95:** Very consistent preferences (or few orders)


Part 5: Scaling Projections
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 25

   * - Scope
     - Single machine (8 cores)
     - Server (64 cores)
   * - Per user (T=365)
     - ~600ms
     - ~600ms
   * - City cohort (100K users)
     - ~3 hours
     - ~30 min
   * - 1% national sample (950K)
     - ~30 hours
     - ~5 hours
   * - Full user base (95M)
     - N/A
     - ~500 hours (cluster)

For truly large-scale deployment, consider:

- **Cloud functions:** Map each user to a Lambda/Cloud Run invocation
- **Spark/Dask:** Distribute across a cluster with ``dask.distributed``
- **Batch scheduling:** Nightly analysis of users who ordered that day
- **Incremental updates:** Only re-analyze users with new orders


Running the Example
-------------------

.. code-block:: bash

   # Default: 100 users, parallel
   python examples/07_uber_eats_scale.py

   # Custom: 500 users, 8 workers
   python examples/07_uber_eats_scale.py --users 500 --workers 8

   # Sequential mode (for profiling)
   python examples/07_uber_eats_scale.py --sequential
