Validation
==========

PyRevealed is validated against the R `revealedPrefs <https://cran.r-project.org/package=revealedPrefs>`_ package,
the standard academic implementation for revealed preference analysis.

Axiom Tests
-----------

All three major axioms (GARP, WARP, SARP) are cross-validated:

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
-----------------------

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
-----------------

The direct revealed preference matrices match exactly:

.. list-table:: Matrix Validation
   :header-rows: 1
   :widths: 40 30 30

   * - Matrix
     - Match
     - Max Diff
   * - Direct prefs (R)
     - OK
     - 0.0000

Performance Comparison
----------------------

PyRevealed is significantly faster due to Numba JIT compilation with parallel execution:

.. list-table:: GARP Performance (T observations, N=10 goods)
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

.. note::

   Performance varies by hardware. Run the validation script for results on your system.

Known Differences
-----------------

**WARP Definition**: PyRevealed and R have slightly different WARP definitions:

- **PyRevealed**: Violation if ``R[i,j] AND P[j,i]`` (strict-weak asymmetry)
- **R revealedPrefs**: Violation if ``R[i,j] AND R[j,i]`` (any mutual preference)

Both are valid interpretations in the literature. Tests avoid cases where definitions diverge.

Function Mapping
----------------

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

Running Validation
------------------

To reproduce the validation on your system:

**1. Install R dependencies:**

.. code-block:: bash

   # Install R (https://cran.r-project.org/)
   # Then in R console:
   install.packages("revealedPrefs")

**2. Install Python dependencies:**

.. code-block:: bash

   pip install rpy2

**3. Run the validation script:**

.. code-block:: bash

   # Full validation (all axioms + efficiency + performance)
   python benchmarks/r_validation.py

   # Quick mode (smaller scale)
   python benchmarks/r_validation.py --quick

   # Correctness only (skip performance)
   python benchmarks/r_validation.py --correctness-only

   # Basic mode (GARP only, legacy output)
   python benchmarks/r_validation.py --basic

Example output:

.. code-block:: text

   ======================================================================
    PYREVEALED vs R revealedPrefs VALIDATION
   ======================================================================

   R and revealedPrefs package detected.

   ======================================================================
    AXIOM TESTS
   ======================================================================
   Axiom    Test Case            PyRevealed   revealedPrefs  Match
   ----------------------------------------------------------------------
   GARP     consistent_3obs      PASS         PASS           OK
   WARP     consistent_3obs      PASS         PASS           OK
   SARP     consistent_3obs      PASS         PASS           OK
   GARP     warp_violation       FAIL         FAIL           OK
   WARP     warp_violation       FAIL         FAIL           OK
   SARP     warp_violation       FAIL         FAIL           OK
   ...

   ======================================================================
    AFRIAT EFFICIENCY TESTS
   ======================================================================
   Efficiency   PyRevealed   revealedPrefs  Match
   --------------------------------------------------
   1.0          FAIL         FAIL           OK
   0.9          FAIL         FAIL           OK
   ...

   ======================================================================
    MATRIX COMPARISON
   ======================================================================
   Matrix               Match      Max Diff
   ----------------------------------------
   Direct prefs (R)     OK         0.0000

   All 16 validation tests PASSED.

See Also
--------

- :doc:`scaling` - Full performance benchmarks
- :doc:`theory` - Mathematical foundations
