Extensions
==========

This section covers extensions to the basic framework: spatial preferences, stochastic choice,
limited attention, and production theory.

General Metric Preferences
--------------------------

Analyze spatial choice with non-Euclidean distance metrics.
Based on Chambers & Echenique (2016) Chapter 11.3-11.4.

Metric Rationality
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^

**Function:** ``determine_best_metric(log)``

Select the metric that best rationalizes the data by minimizing violations:

.. math::

   d^* = \arg\min_{d \in \mathcal{D}} \text{violations}(d)

**Reference:** Chambers & Echenique (2016) Ch. 11


Stochastic Choice
-----------------

Analyze probabilistic choice data with random utility models.
Based on Chambers & Echenique (2016) Chapter 13.

Random Utility Model
^^^^^^^^^^^^^^^^^^^^

**Function:** ``fit_random_utility_model(log)``

In a **Random Utility Model (RUM)**, utility has a deterministic and stochastic component:

.. math::

   U_i = V_i + \varepsilon_i

where :math:`V_i` is the systematic utility and :math:`\varepsilon_i` is a random shock.

**Choice Probability:**

.. math::

   P(i | A) = \Pr\left[ V_i + \varepsilon_i > V_j + \varepsilon_j \, \forall j \in A \right]

Logit Model
^^^^^^^^^^^

**Function:** ``fit_luce_model(log)``

With Type I Extreme Value errors, choice probabilities follow the **logit** form:

.. math::

   P(i | A) = \frac{\exp(V_i)}{\sum_{j \in A} \exp(V_j)}

This is equivalent to Luce's choice axiom.

McFadden's Axioms
^^^^^^^^^^^^^^^^^

**Function:** ``test_mcfadden_axioms(log)``

RUM-consistent choice must satisfy:

1. **Regularity:** :math:`P(i | A) \geq P(i | B)` if :math:`A \subseteq B` and :math:`i \in A`

2. **IIA (for logit):** :math:`\frac{P(i | A)}{P(j | A)} = \frac{P(i | B)}{P(j | B)}` if :math:`i, j \in A \cap B`

.. admonition:: McFadden's Theorem
   :class: important

   Choice probabilities are consistent with RUM if and only if they satisfy a set of
   linear inequalities (Block-Marschak conditions).

**Reference:** McFadden (1974), Block & Marschak (1960), Chambers & Echenique (2016) Ch. 13


Limited Attention
-----------------

Model choice under consideration set constraints.
Based on Chambers & Echenique (2016) Chapter 14.

Attention Filter
^^^^^^^^^^^^^^^^

**Function:** ``test_attention_rationality(log)``

A decision-maker has **limited attention** if they maximize utility over a consideration set:

.. math::

   c(A) = \arg\max_{x \in \Gamma(A)} u(x)

where :math:`\Gamma(A) \subseteq A` is the **consideration set** (items actually considered).

**Interpretation:** Choices that violate SARP might be rationalizable if the decision-maker
doesn't consider all available options.

Consideration Set Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``estimate_consideration_sets(log)``

Estimate the consideration sets :math:`\Gamma(A)` that rationalize observed choices:

.. math::

   \Gamma^*(A) = \min \{ \Gamma : c(A) = \arg\max_{x \in \Gamma} u(x) \text{ for some } u \}

**Axiom (WARP for Attention):**

If :math:`c(A) = x` and :math:`x, y \in B` with :math:`c(B) = y`, then :math:`x \notin \Gamma(B)`.

Salience Weights
^^^^^^^^^^^^^^^^

**Function:** ``compute_salience_weights(log)``

Estimate how likely each item is to enter the consideration set:

.. math::

   \sigma_i = \Pr[i \in \Gamma(A) | i \in A]

Higher salience items are more likely to be considered.

**Reference:** Masatlioglu, Nakajima & Ozbay (2012), Chambers & Echenique (2016) Ch. 14


Production Theory
-----------------

Apply revealed preference to firm behavior.
Based on Chambers & Echenique (2016) Chapter 15.

Profit Maximization
^^^^^^^^^^^^^^^^^^^

**Function:** ``test_profit_maximization(log)``

Define input vector :math:`x^t`, output vector :math:`y^t`, input prices :math:`w^t`, and output prices :math:`p^t`.

**Profit GARP:** No profit arbitrage cycles exist:

.. math::

   \pi^s \geq \pi^t(x^s, y^s) \implies \pi^t \geq \pi^s(x^t, y^t)

where :math:`\pi^t = p^t \cdot y^t - w^t \cdot x^t` is observed profit.

**Weak Axiom of Profit Maximization:**

.. math::

   p^s \cdot y^s - w^s \cdot x^s \geq p^s \cdot y^t - w^s \cdot x^t \implies
   p^t \cdot y^t - w^t \cdot x^t \geq p^t \cdot y^s - w^t \cdot x^s

Cost Minimization
^^^^^^^^^^^^^^^^^

**Function:** ``check_cost_minimization(log)``

For a given output level :math:`y`, cost minimization requires:

.. math::

   w^t \cdot x^t \leq w^t \cdot x^s \text{ whenever } f(x^t) = f(x^s) = y

**GARP for Cost:**

.. math::

   w^s \cdot x^s \leq w^s \cdot x^t \implies w^t \cdot x^t \leq w^t \cdot x^s

Returns to Scale
^^^^^^^^^^^^^^^^

**Function:** ``estimate_returns_to_scale(log)``

Estimate the production technology's returns to scale:

.. math::

   \text{RTS} = \frac{\partial \log y}{\partial \log x}

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - RTS
     - Interpretation
   * - > 1
     - Increasing returns to scale (economies of scale)
   * - = 1
     - Constant returns to scale
   * - < 1
     - Decreasing returns to scale (diseconomies of scale)

**Reference:** Varian (1984), Chambers & Echenique (2016) Ch. 15
