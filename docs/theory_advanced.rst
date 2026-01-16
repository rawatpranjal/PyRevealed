Advanced Topics
===============

This section covers integrability conditions, welfare analysis, additive separability,
and compensated demand.

Integrability Conditions
------------------------

The integrability problem asks whether observed demand can be derived from utility maximization.
Based on Chambers & Echenique (2016) Chapter 6.4-6.5.

Slutsky Matrix
^^^^^^^^^^^^^^

**Function:** ``compute_slutsky_matrix(log)``

The Slutsky matrix captures compensated demand responses:

.. math::

   S_{ij} = \frac{\partial h_i}{\partial p_j}

where :math:`h_i(p, u)` is the Hicksian (compensated) demand for good :math:`i`.

**Slutsky Equation:**

.. math::

   S_{ij} = \frac{\partial x_i}{\partial p_j} + x_j \frac{\partial x_i}{\partial m}

where the first term is the substitution effect and the second is the income effect.

Integrability Test
^^^^^^^^^^^^^^^^^^

**Function:** ``test_integrability(log)``

For demand to be integrable (derivable from utility maximization), the Slutsky matrix must satisfy:

1. **Symmetry:** :math:`S_{ij} = S_{ji}` for all :math:`i, j`

2. **Negative Semi-Definiteness:** :math:`v^\top S v \leq 0` for all vectors :math:`v`

3. **Homogeneity of degree zero:** :math:`\sum_j S_{ij} p_j = 0`

.. admonition:: Hurwicz-Uzawa Theorem
   :class: important

   A differentiable demand function satisfying Walras' Law is integrable if and only if
   the Slutsky matrix is symmetric and negative semi-definite.

**Reference:** Hurwicz & Uzawa (1971), Chambers & Echenique (2016) Ch. 6


Welfare Analysis
----------------

Measure welfare changes from price variations. Based on Chambers & Echenique (2016) Chapter 7.3-7.4.

Compensating Variation
^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_compensating_variation(log)``

The **compensating variation** is the income change that keeps utility constant at the *old* level
after prices change:

.. math::

   CV = e(p^1, u^0) - e(p^0, u^0)

where :math:`e(p, u)` is the expenditure function (minimum cost to achieve utility :math:`u` at prices :math:`p`).

**Interpretation:** How much would we need to compensate the consumer to maintain their old welfare?

Equivalent Variation
^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_equivalent_variation(log)``

The **equivalent variation** is the income change that would produce the same utility change at the *old* prices:

.. math::

   EV = e(p^1, u^1) - e(p^0, u^1)

**Interpretation:** What income change at old prices would be equivalent to the price change?

.. note::

   For infinitesimal price changes, :math:`CV \approx EV \approx` consumer surplus change.
   For discrete changes, they generally differ unless preferences are quasilinear.

**Reference:** Hicks (1939), Chambers & Echenique (2016) Ch. 7


Additive Separability
---------------------

Test whether utility has the additively separable form.
Based on Chambers & Echenique (2016) Chapter 9.3.

Additive Utility
^^^^^^^^^^^^^^^^

**Function:** ``test_additive_separability(log)``

Preferences are **additively separable** if utility can be written as:

.. math::

   U(x_1, \ldots, x_n) = \sum_{i=1}^{n} u_i(x_i)

**Implications:**

1. **No cross-price effects:** :math:`\frac{\partial^2 U}{\partial x_i \partial x_j} = 0` for :math:`i \neq j`

2. **Independent valuations:** Marginal utility of good :math:`i` depends only on :math:`x_i`

3. **Constant marginal rate of substitution:** MRS between two goods depends only on their quantities

Testable Restriction
^^^^^^^^^^^^^^^^^^^^

**Function:** ``check_no_cross_effects(log)``

For additively separable preferences, the cross-price effect should be entirely due to income effects:

.. math::

   \frac{\partial x_i}{\partial p_j} = -x_j \frac{\partial x_i}{\partial m} \quad \text{for } i \neq j

**Reference:** Chambers & Echenique (2016) Ch. 9


Compensated Demand
------------------

Decompose price effects into substitution and income components.
Based on Chambers & Echenique (2016) Chapter 10.3.

Slutsky Decomposition
^^^^^^^^^^^^^^^^^^^^^

**Function:** ``decompose_price_effects(log)``

The total effect of a price change decomposes into substitution and income effects:

.. math::

   \underbrace{\frac{\partial x_i}{\partial p_j}}_{\text{total effect}} =
   \underbrace{\frac{\partial h_i}{\partial p_j}}_{\text{substitution effect}} -
   \underbrace{x_j \frac{\partial x_i}{\partial m}}_{\text{income effect}}

Hicksian Demand
^^^^^^^^^^^^^^^

**Function:** ``compute_hicksian_demand(log)``

The **Hicksian demand** function :math:`h(p, u)` gives the cheapest bundle achieving utility :math:`u`:

.. math::

   h(p, u) = \arg\min_x \{ p \cdot x : U(x) \geq u \}

**Properties:**

1. **Homogeneous of degree zero** in prices
2. **Shephard's Lemma:** :math:`h_i(p, u) = \frac{\partial e(p, u)}{\partial p_i}`
3. **Compensated Law of Demand:** :math:`(p' - p) \cdot (h(p', u) - h(p, u)) \leq 0`

**Reference:** Hicks (1939), Chambers & Echenique (2016) Ch. 10
