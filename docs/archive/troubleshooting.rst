Troubleshooting
===============

This guide helps diagnose and resolve common errors in PrefGraph.


Exception Reference
-------------------

PrefGraph provides a hierarchy of custom exceptions that inherit from
``ValueError`` for backward compatibility. You can catch specific exception
types or use the base ``PrefGraphError`` to catch all library errors.

.. code-block:: python

   from prefgraph import PrefGraphError, NaNInfError

   try:
       log = BehaviorLog(prices, quantities)
   except NaNInfError as e:
       print(f"Data has missing values: {e}")
   except PrefGraphError as e:
       print(f"PrefGraph error: {e}")
   except ValueError as e:
       # Also catches PrefGraphError (for backward compatibility)
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
   from prefgraph import BehaviorLog

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
   from prefgraph import BehaviorLog

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

.. code-block:: python

   import numpy as np
   from prefgraph import BehaviorLog

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

   from prefgraph import BehaviorLog, fit_latent_values, build_value_function

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

   from prefgraph import compute_integrity_score

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

   from prefgraph import PreferenceEncoder, BehaviorLog

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

PrefGraph provides flexible NaN handling via the ``nan_policy`` parameter
available on ``BehaviorLog`` and other data containers.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50
