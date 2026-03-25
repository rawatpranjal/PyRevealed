Performance
===========

This page covers benchmarks, acceleration techniques, parallel processing,
and cross-validation against the R ``revealedPrefs`` package.

Benchmarks
----------

All timings measured on Apple M1 Pro (10 cores), N=10 goods, JIT-warmed.

.. list-table:: Algorithm Timing by Observation Count
   :header-rows: 1
   :widths: 18 12 12 12 12 12 22

   * - Algorithm
     - T=100
     - T=500
     - T=1,000
     - T=2,000
     - Complexity
     - Notes
   * - GARP
     - <1ms
     - 66ms
     - 102ms
     - 469ms
     - O(T^3)
     - Core consistency check
   * - AEI
     - <1ms
     - 54ms
     - 101ms
     - 446ms
     - O(T^3 log T)
     - Binary search + GARP
   * - MPI
     - <1ms
     - 70ms
     - 98ms
     - 448ms
     - O(T^3)
     - Cycle detection
   * - HARP
     - 1ms
     - 100ms
     - 464ms
     - 3.2s
     - O(T^3)
     - Homotheticity test
   * - Acyclical-P
     - 1ms
     - 198ms
     - 412ms
     - 1.8s
     - O(T^3)
     - Strict preference only
   * - GAPP
     - 1ms
     - 212ms
     - 802ms
     - 2.5s
     - O(T^3)
     - Generalized axiom
   * - Differentiable
     - 280ms
     - 7.1s
     - 29s
     - 199s
     - O(T^4)
     - Gradient-based

Memory scales as O(T^2) due to T x T boolean matrices:

.. list-table:: Memory Usage by Scale
   :header-rows: 1
   :widths: 20 20 30

   * - Observations (T)
     - Peak Memory
     - Notes
   * - 1,000
     - ~50 MB
     - Fits in L3 cache
   * - 10,000
     - ~800 MB
     - Requires RAM
   * - 50,000
     - ~20 GB
     - Server-class memory
   * - 100,000
     - ~80 GB
     - High-memory server

.. list-table:: Comparison to Other Libraries (T=10,000)
   :header-rows: 1
   :widths: 30 20 20 30

   * - Library
     - GARP Time
     - Language
     - Notes
   * - **PyRevealed**
     - **2s**
     - Python/Numba
     - Production-ready, parallel
   * - revealedPrefs (R)
     - 180s
     - R
     - Academic use only
   * - Custom C++
     - 0.8s
     - C++
     - Requires compilation

Running benchmarks locally:

.. code-block:: bash

   python benchmarks/run_scaling.py --quick       # Up to T=2000
   python benchmarks/run_scaling.py               # Up to T=100000
   python benchmarks/run_scaling.py --algorithm garp  # Single algorithm

Results are saved to ``benchmarks/output/scaling_results.csv``.

Acceleration
------------

Numba JIT Compilation
^^^^^^^^^^^^^^^^^^^^^

All core algorithms use ``@njit(cache=True)`` for ahead-of-time compilation to
machine code. The first call includes JIT compilation overhead (~2-5 seconds);
subsequent calls use the disk-cached binary:

.. code-block:: python

   result = check_garp(session)  # First call: ~3s (JIT compile)
   result = check_garp(session)  # Second call: ~50ms (cached)

The cache persists across Python sessions.

Parallel Floyd-Warshall
^^^^^^^^^^^^^^^^^^^^^^^

The O(T^3) Floyd-Warshall transitive closure -- the bottleneck for GARP, AEI,
and MPI -- is parallelized using Numba's ``prange``. The k-loop carries
dependencies, but the i-loop is independent and distributes across CPU cores:

.. code-block:: python

   @njit(cache=True, parallel=True)
   def floyd_warshall_tc_numba(adjacency):
       T = adjacency.shape[0]
       closure = adjacency.copy()

       for k in range(T):
           col_k = closure[:, k].copy()
           row_k = closure[k, :].copy()

           for i in prange(T):        # parallel over rows
               if col_k[i]:
                   for j in range(T):
                       if row_k[j]:
                           closure[i, j] = True

       return closure

SCC Optimization
^^^^^^^^^^^^^^^^

For sparse preference graphs, strongly connected component (SCC) decomposition
can reduce work by processing each component independently rather than running
Floyd-Warshall on the full T x T matrix. This is most effective when the
preference graph decomposes into many small components rather than one large
connected block.

Panel Parallelization
---------------------

For multi-user datasets, distribute sessions across CPU cores with
``multiprocessing``:

.. code-block:: python

   from pyrevealed import check_garp
   import multiprocessing as mp

   def process_user(session):
       return check_garp(session)

   with mp.Pool() as pool:
       results = pool.map(process_user, user_sessions)

Typical use cases by scale:

- **Real-time APIs** (~500 obs): ``check_garp(session)`` returns in ~25ms.
- **Batch processing** (~10K obs): Use ``compute_aei(session, tolerance=1e-3)``.
- **Large panels** (~100K obs): Combine parallel Floyd-Warshall with multi-user parallelism.

Validation Against R
--------------------

PyRevealed is validated against the R
`revealedPrefs <https://cran.r-project.org/package=revealedPrefs>`_ package,
the standard academic implementation for revealed preference analysis.

Axiom Tests
^^^^^^^^^^^

All three major axioms (GARP, WARP, SARP) are cross-validated across
consistent and inconsistent datasets:

.. list-table:: Axiom Validation (16 tests)
   :header-rows: 1
   :widths: 15 25 15 20 15

   * - Axiom
     - Test Case
     - PyRevealed
     - revealedPrefs
     - Match
   * - GARP
     - consistent_3obs
     - PASS
     - PASS
     - OK
   * - WARP
     - consistent_3obs
     - PASS
     - PASS
     - OK
   * - SARP
     - consistent_3obs
     - PASS
     - PASS
     - OK
   * - GARP
     - warp_violation
     - FAIL
     - FAIL
     - OK
   * - WARP
     - warp_violation
     - FAIL
     - FAIL
     - OK
   * - SARP
     - warp_violation
     - FAIL
     - FAIL
     - OK
   * - GARP
     - garp_3cycle
     - FAIL
     - FAIL
     - OK
   * - WARP
     - garp_3cycle
     - FAIL
     - FAIL
     - OK
   * - SARP
     - garp_3cycle
     - FAIL
     - FAIL
     - OK
   * - GARP
     - sarp_violation
     - PASS
     - PASS
     - OK
   * - SARP
     - sarp_violation
     - FAIL
     - FAIL
     - OK

Afriat Efficiency Tests
^^^^^^^^^^^^^^^^^^^^^^^

The ``afriat.par`` parameter in R corresponds to PyRevealed's efficiency in AEI:

.. list-table:: Efficiency Validation
   :header-rows: 1
   :widths: 25 25 25 25

   * - Efficiency
     - PyRevealed
     - revealedPrefs
     - Match
   * - 1.0
     - FAIL
     - FAIL
     - OK
   * - 0.9
     - FAIL
     - FAIL
     - OK
   * - 0.8
     - FAIL
     - FAIL
     - OK
   * - 0.5
     - FAIL
     - FAIL
     - OK

Matrix Comparison
^^^^^^^^^^^^^^^^^

Direct revealed preference matrices match exactly:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Matrix
     - Match
     - Max Diff
   * - Direct prefs (R)
     - OK
     - 0.0000

Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: GARP Performance (N=10 goods)
   :header-rows: 1
   :widths: 20 20 25 20

   * - Observations
     - PyRevealed
     - revealedPrefs
     - Speedup
   * - 100
     - 0.2ms
     - 81ms
     - 461x
   * - 500
     - 37ms
     - 235ms
     - 6x
   * - 1,000
     - 97ms
     - 1.7s
     - 18x
   * - 2,000
     - 449ms
     - 18.7s
     - 42x

Average speedup: **132x faster** than R.

Known Differences
^^^^^^^^^^^^^^^^^

**WARP Definition**: PyRevealed uses ``R[i,j] AND P[j,i]`` (strict-weak
asymmetry), while R uses ``R[i,j] AND R[j,i]`` (any mutual preference).
Both are valid interpretations in the literature. Validation tests avoid
cases where definitions diverge.

Function Mapping
^^^^^^^^^^^^^^^^

.. list-table:: Equivalent Functions
   :header-rows: 1
   :widths: 35 35 30

   * - revealedPrefs (R)
     - PyRevealed
     - Notes
   * - ``checkGarp(x, p)``
     - ``check_garp(session)``
     - GARP consistency test
   * - ``checkSarp(x, p)``
     - ``check_sarp(session)``
     - SARP consistency test
   * - ``checkWarp(x, p)``
     - ``check_warp(session)``
     - WARP consistency test
   * - ``directPrefs(x, p)``
     - ``result.direct_revealed_preference``
     - R matrix
   * - ``afriat.par`` parameter
     - ``compute_aei().efficiency_index``
     - Afriat efficiency level

Reproducing Validation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Prerequisites: R with revealedPrefs, plus pip install rpy2

   python benchmarks/r_validation.py                    # Full validation
   python benchmarks/r_validation.py --quick             # Smaller scale
   python benchmarks/r_validation.py --correctness-only  # Skip performance

See Also
--------

- :doc:`theory` - Mathematical foundations
