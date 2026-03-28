Structural Preference Analysis and Utility Recovery
==================================================

This section delineates axiomatic tests for specific preference structures, including homotheticity, quasilinearity, and separability, as well as the formal methodology for utility recovery.

Homothetic Preferences (HARP)
-----------------------------

**Reference Implementation:** ``validate_proportional_scaling(log)``

The Homothetic Axiom of Revealed Preference (HARP) evaluates whether an agent's preferences are invariant to proportional scaling of income, implying that commodity demand scales linearly with total expenditure.

**Formal Definition:**

Define the expenditure ratio :math:`r_{ij}` as the cost of bundle :math:`j` evaluated at prices :math:`i` relative to the actual expenditure at observation :math:`i`:

.. math::

   r_{ij} = \frac{p^i \cdot x^i}{p^i \cdot x^j}

**The HARP Condition:**

.. math::

   \text{HARP is satisfied} \iff \nexists \text{ cycle } i_1 \to i_2 \to \cdots \to i_m \to i_1 : \prod_{k=1}^{m} r_{i_k, i_{k+1}} > 1

Equivalently, in logarithmic space:

.. math::

   \sum_{k=1}^{m} \log r_{i_k, i_{k+1}} \leq 0

**Reference:** Varian (1983).


Quasilinear Utility (Income Invariance)
---------------------------------------

**Reference Implementation:** ``test_income_invariance(log)``

Quasilinearity implies a utility function of the form :math:`U(x, m) = v(x) + m`, where the demand for commodity :math:`x` is independent of the agent's income level :math:`m`. This is evaluated via the condition of cyclic monotonicity.

**The Quasilinearity Condition:**

For any sequence of observations forming a cycle :math:`i_1 \to i_2 \to \cdots \to i_m \to i_1`, the following must hold:

.. math::

   \sum_{k=1}^{m} p^{i_k} \cdot (x^{i_{k+1}} - x^{i_k}) \geq 0

**Behavioral Interpretation:**

A failure of quasilinearity suggests that the agent's marginal utility of income is not constant, and choices are influenced by income effects rather than relative prices alone.

**Reference:** Rochet (1987).


Weak Separability (Feature Independence)
----------------------------------------

**Reference Implementation:** ``test_feature_independence(log, group_a, group_b)``

Weak separability posits that preferences over a subset of commodities (Group A) are independent of the consumption levels of another subset (Group B). Formally, :math:`U(x_A, x_B) = V(u_A(x_A), u_B(x_B))`.

**Analytical Heuristic:**

The implementation evaluates separability by examining the consistency (CCEI) of choices within partitioned commodity groups and assessing the degree of cross-group correlation.

**Reference:** Chambers & Echenique (2016).


Utility Recovery via Afriat’s Inequalities
------------------------------------------

**Reference Implementation:** ``fit_latent_values(log)``

If the observed data satisfy GARP, Afriat's Theorem guarantees the existence of a continuous, monotonic, and concave utility function that rationalizes the behavior. PrefGraph recovers the latent utility values :math:`U_k` and marginal utilities of income (Lagrange multipliers) :math:`\lambda_k > 0`.

**Linear Programming Formulation:**

The recovery is achieved by solving a system of Afriat inequalities for all observation pairs :math:`(k, l)`:

.. math::

   U_k \leq U_l + \lambda_l \cdot p^l \cdot (x^k - x^l) \quad \forall \, k, l

**Optimization Objective:**

.. math::

   \min \sum_{k=1}^{T} \lambda_k

The resulting utility function is the lower envelope of the recovered tangent planes, providing a piecewise linear and concave approximation of the agent's true preferences.

**References:** Afriat (1967), Varian (1982).
