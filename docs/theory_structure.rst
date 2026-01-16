Preference Structure
====================

This section covers tests for specific preference structures and utility recovery.

Proportional Scaling (HARP)
---------------------------

**Function:** ``validate_proportional_scaling(log)``

Tests homothetic preferences (demand scales proportionally with income).

Define expenditure ratio:

.. math::

   r_{ij} = \frac{p^i \cdot x^i}{p^i \cdot x^j}

**HARP Condition:**

.. math::

   \text{HARP holds} \iff \text{no cycle } i_1 \to i_2 \to \cdots \to i_m \to i_1 \text{ with } \prod_{k=1}^{m} r_{i_k, i_{k+1}} > 1

In log-space (Floyd-Warshall):

.. math::

   \sum_{k=1}^{m} \log r_{i_k, i_{k+1}} \leq 0

**Reference:** Varian (1983)


Income Invariance (Quasilinearity)
----------------------------------

**Function:** ``test_income_invariance(log)``

Tests quasilinear utility :math:`U(x, m) = v(x) + m` via cyclic monotonicity.

**Condition:** For any cycle :math:`i_1 \to i_2 \to \cdots \to i_m \to i_1`:

.. math::

   \sum_{k=1}^{m} p^{i_k} \cdot (x^{i_{k+1}} - x^{i_k}) \geq 0

**Interpretation:** Choices depend only on relative prices, not income level.

**Reference:** Rochet (1987)


Feature Independence (Separability)
-----------------------------------

**Function:** ``test_feature_independence(log, group_a, group_b)``

Tests weak separability: :math:`U(x_A, x_B) = V(u_A(x_A), u_B(x_B))`

**Heuristic Test:**

1. Compute AEI within each group
2. Measure cross-group correlation
3. Separable if :math:`\text{AEI}_A > 0.9`, :math:`\text{AEI}_B > 0.9`, and cross-effect < 0.2

**Reference:** Chambers & Echenique (2016) Ch. 4, Theorem 4.4


Cross-Price Effects
-------------------

**Function:** ``test_cross_price_effect(log, good_g, good_h)``

**Gross Substitutes:**

.. math::

   \Delta p_g > 0, \, \Delta x_g < 0 \implies \Delta x_h > 0

**Gross Complements:**

.. math::

   \Delta p_g > 0, \, \Delta x_g < 0 \implies \Delta x_h < 0

**Reference:** Hicks (1939)


Utility Recovery
----------------

Afriat's Inequalities
^^^^^^^^^^^^^^^^^^^^^

**Function:** ``fit_latent_values(log)``

If GARP holds, there exist utility values :math:`U_k` and Lagrange multipliers :math:`\lambda_k > 0` satisfying:

.. math::

   U_k \leq U_l + \lambda_l \cdot p^l \cdot (x^k - x^l) \quad \forall \, k, l

**Linear Program:**

- Variables: :math:`U_1, \ldots, U_T, \lambda_1, \ldots, \lambda_T`
- Constraints: Afriat inequalities for all :math:`(k, l)` pairs
- Objective: Minimize :math:`\sum_k \lambda_k`

The recovered utility is piecewise linear and concave.

**Reference:** Afriat (1967), Varian (1982)
