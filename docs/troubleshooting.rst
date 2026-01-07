Troubleshooting
===============

This guide helps diagnose and resolve common errors in PyRevealed.


Exception Reference
-------------------

PyRevealed provides a hierarchy of custom exceptions that inherit from
``ValueError`` for backward compatibility. You can catch specific exception
types or use the base ``PyRevealedError`` to catch all library errors.

.. code-block:: python

   from pyrevealed import PyRevealedError, NaNInfError

   try:
       log = BehaviorLog(prices, quantities)
   except NaNInfError as e:
       print(f"Data has missing values: {e}")
   except PyRevealedError as e:
       print(f"PyRevealed error: {e}")
   except ValueError as e:
       # Also catches PyRevealedError (for backward compatibility)
       print(f"Validation error: {e}")


DataValidationError
^^^^^^^^^^^^^^^^^^^

Raised when input data fails validation checks. This is the base class for
more specific validation errors.

**Subclasses:**

- ``DimensionError``: Array shapes don't match or wrong dimensions
- ``ValueRangeError``: Values outside expected ranges
- ``NaNInfError``: NaN or Inf values detected


DimensionError
^^^^^^^^^^^^^^

Raised when array dimensions are incompatible.

**Common causes:**

* ``cost_vectors`` and ``action_vectors`` have different shapes
* Arrays are not 2D (should be T observations x N features)
* Empty arrays (T=0 or N=0)

**Example:**

.. code-block:: python

   import numpy as np
   from pyrevealed import BehaviorLog

   prices = np.array([[1, 2, 3]])      # shape (1, 3)
   quantities = np.array([[1, 2]])     # shape (1, 2) - mismatch!

   # Raises DimensionError:
   # "cost_vectors shape (1, 3) does not match action_vectors shape (1, 2).
   #  Both arrays must have shape (T, N) where T=observations and N=features.
   #  Hint: Check that your price and quantity data have the same dimensions."

**Solution:**

.. code-block:: python

   # Check shapes before creating BehaviorLog
   print(f"Prices shape: {prices.shape}")
   print(f"Quantities shape: {quantities.shape}")

   # Ensure shapes match
   assert prices.shape == quantities.shape


ValueRangeError
^^^^^^^^^^^^^^^

Raised when values are outside expected ranges.

**Common causes:**

* Negative or zero prices/costs
* Negative quantities/actions
* Probabilities outside [0, 1]

**Example:**

.. code-block:: python

   import numpy as np
   from pyrevealed import BehaviorLog

   prices = np.array([[1, -2, 3], [2, 1, 0]])  # Negative and zero prices!
   quantities = np.array([[1, 2, 3], [2, 1, 1]])

   # Raises ValueRangeError:
   # "Found 2 non-positive costs at positions: [[0, 1], [1, 2]].
   #  All costs must be strictly positive (> 0) for revealed preference analysis.
   #  Hint: Check for missing data encoded as 0, or filter out zero-cost observations."

**Solution:**

.. code-block:: python

   # Check for invalid values
   print(f"Non-positive prices: {(prices <= 0).sum()}")
   print(f"Negative quantities: {(quantities < 0).sum()}")

   # Filter out invalid rows
   valid_mask = (prices > 0).all(axis=1) & (quantities >= 0).all(axis=1)
   prices = prices[valid_mask]
   quantities = quantities[valid_mask]


NaNInfError
^^^^^^^^^^^

Raised when NaN or Inf values are detected in input data.

**Common causes:**

* Missing data encoded as NaN
* Division by zero in preprocessing
* Numeric overflow producing Inf

**Example:**

.. code-block:: python

   import numpy as np
   from pyrevealed import BehaviorLog

   prices = np.array([[1, np.nan, 3], [2, 1, 1]])
   quantities = np.array([[1, 2, 3], [2, 1, 1]])

   # Default behavior - raises NaNInfError:
   # "Found 1 NaN/Inf values in 1 observations. Affected rows: [0].
   #  Use nan_policy='drop' to remove affected rows, or
   #  nan_policy='warn' to drop with a warning."

**Solution:**

Use the ``nan_policy`` parameter to handle NaN values automatically:

.. code-block:: python

   # Option 1: Clean data manually before
   prices = np.nan_to_num(prices, nan=prices[np.isfinite(prices)].mean())

   # Option 2: Use nan_policy='drop' to remove rows with NaN
   log = BehaviorLog(prices, quantities, nan_policy='drop')

   # Option 3: Use nan_policy='warn' to drop with a warning
   log = BehaviorLog(prices, quantities, nan_policy='warn')


OptimizationError
^^^^^^^^^^^^^^^^^

Raised when an optimization solver fails to find a solution.

**Common causes:**

* Data is too inconsistent for utility recovery
* Linear programming constraints are infeasible
* Numerical issues prevent convergence

**Example:**

.. code-block:: python

   from pyrevealed import BehaviorLog, fit_latent_values, build_value_function

   log = BehaviorLog(prices, quantities)
   result = fit_latent_values(log)

   # If result.success is False, building value function will fail:
   value_fn = build_value_function(log, result)
   # Raises OptimizationError:
   # "Cannot construct utility from failed recovery. LP status: ...
   #  Hint: Check data consistency with compute_integrity_score() first.
   #  If integrity is low, the behavior may be too inconsistent for utility recovery."

**Solution:**

.. code-block:: python

   from pyrevealed import compute_integrity_score

   # Check consistency first
   integrity = compute_integrity_score(log)
   print(f"Integrity score: {integrity.efficiency_index}")

   if integrity.efficiency_index < 0.7:
       print("Data is too inconsistent for utility recovery")
   else:
       result = fit_latent_values(log)
       if result.success:
           value_fn = build_value_function(log, result)


NotFittedError
^^^^^^^^^^^^^^

Raised when an operation requires a fitted model.

**Example:**

.. code-block:: python

   from pyrevealed import PreferenceEncoder, BehaviorLog

   encoder = PreferenceEncoder()

   # Forgot to call fit() first!
   features = encoder.extract_latent_values()
   # Raises NotFittedError:
   # "Encoder not fitted. Call fit() first, or check if behavior
   #  is too inconsistent (use BehavioralAuditor to check).
   #  Hint: Use compute_integrity_score() to check data consistency before fitting."

**Solution:**

.. code-block:: python

   encoder = PreferenceEncoder()
   encoder.fit(log)  # Fit first!
   features = encoder.extract_latent_values()


InsufficientDataError
^^^^^^^^^^^^^^^^^^^^^

Raised when there is not enough data for the requested operation.

**Common causes:**

* Only 1 observation (need at least 2 for comparisons)
* Choice sets with only 1 item
* Groups too small for separability testing

**Solution:**

Ensure you have enough observations for the analysis you want to perform.


Handling NaN Values
-------------------

PyRevealed provides flexible NaN handling via the ``nan_policy`` parameter
available on ``BehaviorLog`` and other data containers.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Policy
     - Behavior
     - Use When
   * - ``"raise"`` (default)
     - Raises ``NaNInfError``
     - You want strict validation
   * - ``"warn"``
     - Warns and drops affected rows
     - You want visibility into data quality
   * - ``"drop"``
     - Silently drops affected rows
     - You trust your preprocessing

**Examples:**

.. code-block:: python

   import numpy as np
   from pyrevealed import BehaviorLog

   # Data with some NaN values
   prices = np.array([[1, 2], [np.nan, 1], [2, 2]])
   quantities = np.array([[3, 1], [1, 3], [2, 2]])

   # Strict mode (default) - will raise if NaN found
   # log = BehaviorLog(prices, quantities)  # NaNInfError!

   # Lenient mode - drops bad rows with warning
   log = BehaviorLog(prices, quantities, nan_policy='warn')
   # Warning: Dropping 1 observations with NaN/Inf values (rows: [1]).
   print(f"Observations after cleaning: {log.num_records}")  # 2

   # Silent mode - drops without warning
   log = BehaviorLog(prices, quantities, nan_policy='drop')


Warning Control
---------------

PyRevealed emits warnings for non-critical issues that may affect results.

**Warning types:**

- ``DataQualityWarning``: Data quality issues (dropped rows, rank-deficiency)
- ``NumericalInstabilityWarning``: Potential numerical issues

**Controlling warnings:**

.. code-block:: python

   import warnings
   from pyrevealed import DataQualityWarning

   # Suppress data quality warnings
   warnings.filterwarnings('ignore', category=DataQualityWarning)

   # Promote to errors (fail fast)
   warnings.filterwarnings('error', category=DataQualityWarning)

   # Reset to default
   warnings.filterwarnings('default', category=DataQualityWarning)


Numerical Stability
-------------------

PyRevealed uses tolerance parameters for floating-point comparisons.

**Default tolerances:**

- GARP checks: ``1e-10``
- AEI binary search: ``1e-6``
- LP solvers: ``1e-8``

**Adjusting tolerance:**

.. code-block:: python

   from pyrevealed import validate_consistency

   # For noisy real-world data, use larger tolerance
   result = validate_consistency(log, tolerance=1e-6)

   # For clean synthetic data, use smaller tolerance
   result = validate_consistency(log, tolerance=1e-12)

**Warning signs of numerical issues:**

- Results change significantly with small tolerance changes
- Optimization converges but with high residuals
- Unexpected GARP violations in clearly consistent data

**Tips for numerical stability:**

1. Scale your data to reasonable ranges (avoid very large or small values)
2. Use consistent units across all observations
3. Consider normalizing prices by income or total expenditure


Common Issues FAQ
-----------------

**Q: My data seems consistent but GARP check fails?**

A: Try increasing the tolerance parameter:

.. code-block:: python

   result = validate_consistency(log, tolerance=1e-6)

If this fixes it, your data has small numerical noise that violates strict
consistency. The tolerance allows for floating-point imprecision.

**Q: PreferenceEncoder won't fit?**

A: Check your data's integrity score first:

.. code-block:: python

   from pyrevealed import compute_integrity_score

   result = compute_integrity_score(log)
   print(f"Integrity: {result.efficiency_index}")

   # If integrity is below ~0.7, utility recovery may fail
   # Consider filtering highly inconsistent observations

**Q: I get different results on different runs?**

A: Check for numerical edge cases:

1. Values very close to zero
2. Nearly collinear price vectors
3. Very small price differences between observations

Try scaling your data or using a more robust tolerance.

**Q: How do I handle missing observations?**

A: Use ``nan_policy`` when creating your BehaviorLog:

.. code-block:: python

   log = BehaviorLog(prices, quantities, nan_policy='drop')

Or clean your data beforehand:

.. code-block:: python

   # Remove rows with any missing values
   valid_mask = np.isfinite(prices).all(axis=1) & np.isfinite(quantities).all(axis=1)
   prices = prices[valid_mask]
   quantities = quantities[valid_mask]
