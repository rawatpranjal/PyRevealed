Efficiency Metrics
==================

When GARP fails, these metrics quantify the severity of violations and the power of the test.

Integrity Score (Afriat Efficiency Index)
-----------------------------------------

**Function:** ``compute_integrity_score(log)``

**Definition:**

.. math::

   \text{AEI} = \sup \left\{ e \in [0,1] : \text{GARP holds with } e \cdot p^i \cdot x^i \geq p^i \cdot x^j \right\}

**Algorithm:** Binary search over :math:`e`. At efficiency :math:`e`, the modified revealed preference becomes:

.. math::

   x^i \, R_e \, x^j \iff e \cdot (p^i \cdot x^i) \geq p^i \cdot x^j

**Interpretation:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - AEI
     - Interpretation
   * - 1.0
     - Perfectly consistent - all choices consistent with utility maximization
   * - 0.95+
     - Minor deviations - Varian's threshold for approximate consistency
   * - 0.85-0.95
     - Moderate deviations - some inconsistent choices
   * - < 0.70
     - Substantial departures from consistency

.. note::

   **Benchmark:** CKMS (2014) found mean AEI = 0.881 in controlled lab experiments.
   E-commerce data typically shows lower values due to measurement noise.

**Reference:** Afriat (1972), Varian (1990)


Confusion Metric (Money Pump Index)
-----------------------------------

**Function:** ``compute_confusion_metric(log)``

For a violation cycle :math:`k_1 \to k_2 \to \cdots \to k_m \to k_1`:

.. math::

   \text{MPI} = \frac{\sum_{i=1}^{m} \left( p^{k_i} \cdot x^{k_i} - p^{k_i} \cdot x^{k_{i+1}} \right)}{\sum_{i=1}^{m} p^{k_i} \cdot x^{k_i}}

**Interpretation:** Maximum fraction of spending extractable by cycling through preference contradictions.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - MPI
     - Interpretation
   * - 0
     - No cycles - fully consistent preferences
   * - 0.01-0.10
     - Minor cycles - small inconsistencies
   * - 0.10-0.30
     - Moderate cycles - noticeable inconsistencies
   * - > 0.30
     - Severe cycles - substantial inconsistencies

**Reference:** Chambers & Echenique (2016) Ch. 5


Outlier Fraction (Houtman-Maks Index)
-------------------------------------

**Function:** ``compute_minimal_outlier_fraction(log)``

.. math::

   \text{HM} = \min \left\{ \frac{|S|}{T} : \text{removing observations } S \text{ makes data GARP-consistent} \right\}

**Reference:** Houtman & Maks (1985)


Per-Observation Efficiency (Varian's Index)
-------------------------------------------

**Function:** ``compute_granular_integrity(log)``

Solve the linear program:

.. math::

   \min \sum_{i=1}^{T} (1 - e_i)

subject to:

.. math::

   e_i \cdot (p^i \cdot x^i) \geq p^i \cdot x^j \quad \forall \, (i,j) : x^i \, R^* \, x^j

.. math::

   0 \leq e_i \leq 1

**Interpretation:** Identifies which specific observations are problematic.

**Reference:** Varian (1990)


Test Power (Bronars' Index)
---------------------------

**Function:** ``compute_test_power(log)``

.. math::

   \text{Power} = \frac{\#\{\text{random behaviors violating GARP}\}}{\#\{\text{simulations}\}}

Random bundles are generated via symmetric Dirichlet distribution on budget hyperplanes:

.. math::

   \text{shares} \sim \text{Dirichlet}(1, 1, \ldots, 1), \quad x^t_j = \frac{\text{share}_j \cdot e_t}{p^t_j}

**Interpretation:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Power
     - Interpretation
   * - > 0.90
     - Excellent - passing GARP is highly informative
   * - 0.70-0.90
     - Good - meaningful test of consistency
   * - 0.50-0.70
     - Moderate - some discrimination but inconclusive
   * - < 0.50
     - Weak - even random behavior often passes; test lacks power

.. warning::

   Low power indicates budget sets don't overlap much. A high AEI with low power
   may just mean the data couldn't detect inconsistency, not that the consumer is consistent.

**Reference:** Bronars (1987)
