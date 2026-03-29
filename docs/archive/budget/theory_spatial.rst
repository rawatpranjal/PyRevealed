General Metric Preferences
==========================

Analyze spatial choice with non-Euclidean distance metrics.
Based on Chambers & Echenique (2016) Chapter 11.3-11.4.

Metric Rationality
------------------

**Function:** ``find_ideal_point_general(log)``

A decision-maker has **metric preferences** if there exists an ideal point :math:`z^*` such that:

.. math::

   x \succ y \iff d(x, z^*) < d(y, z^*)

where :math:`d(\cdot, \cdot)` is a distance metric.

**Supported Metrics:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Formula
     - Use Case
   * - Euclidean
     - :math:`\|x - z^*\|_2`
     - Standard spatial preferences
   * - Manhattan
     - :math:`\|x - z^*\|_1`
     - Grid-like feature spaces
   * - Chebyshev
     - :math:`\|x - z^*\|_\infty`
     - Worst-case aversion
   * - Minkowski
     - :math:`\|x - z^*\|_p`
     - Generalized norms

Metric Selection
----------------

**Function:** ``determine_best_metric(log)``

Select the metric that best rationalizes the data by minimizing violations:

.. math::

   d^* = \arg\min_{d \in \mathcal{D}} \text{violations}(d)

**Reference:** Chambers & Echenique (2016) Ch. 11
