Mathematical Foundations
========================

This page provides formal mathematical definitions for all methods in PyRevealed.

Notation
--------

.. list-table::
   :widths: 20 80

   * - :math:`p^t \in \mathbb{R}^n_+`
     - Price vector at observation :math:`t`
   * - :math:`x^t \in \mathbb{R}^n_+`
     - Quantity vector (bundle) chosen at observation :math:`t`
   * - :math:`e_t = p^t \cdot x^t`
     - Expenditure at observation :math:`t`
   * - :math:`E_{ij} = p^i \cdot x^j`
     - Cost of bundle :math:`j` at prices :math:`i`
   * - :math:`T`
     - Number of observations
   * - :math:`n`
     - Number of goods

---

Consistency Tests
-----------------

GARP (Generalized Axiom of Revealed Preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_consistency(log)``

**Revealed Preference Relations:**

Define weak revealed preference :math:`R` and strict revealed preference :math:`P`:

.. math::

   x^i \, R \, x^j \iff p^i \cdot x^i \geq p^i \cdot x^j

.. math::

   x^i \, P \, x^j \iff p^i \cdot x^i > p^i \cdot x^j

Let :math:`R^*` be the transitive closure of :math:`R` (computed via Floyd-Warshall).

**GARP Condition:**

.. math::

   \text{GARP holds} \iff \nexists \, i,j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, P \, x^i \right)

**Interpretation:** If bundle :math:`i` is transitively revealed preferred to bundle :math:`j`, then :math:`j` cannot be strictly revealed preferred to :math:`i`.

**Reference:** Varian (1982), Chambers & Echenique (2016) Ch. 3

---

WARP (Weak Axiom of Revealed Preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_consistency_weak(log)``

**WARP Condition:**

.. math::

   \text{WARP holds} \iff \nexists \, i,j : \left( x^i \, R \, x^j \right) \land \left( x^j \, P \, x^i \right)

Unlike GARP, WARP only checks direct (length-2) violations without transitivity.

**Reference:** Samuelson (1938), Chambers & Echenique (2016) Ch. 2

---

SARP (Strict Axiom of Revealed Preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_sarp(log)``

**SARP Condition (Antisymmetry):**

.. math::

   \text{SARP holds} \iff \nexists \, i \neq j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, R^* \, x^i \right)

SARP prohibits indifference cycles. Stronger than GARP.

**Reference:** Chambers & Echenique (2016) Ch. 2

---

Smooth Preferences (Differentiable Rationality)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_smooth_preferences(log)``

Two conditions must hold:

1. **SARP:** No indifference cycles
2. **Price-Quantity Uniqueness:**

.. math::

   p^t \neq p^s \implies x^t \neq x^s

**Interpretation:** Demand function is well-defined and differentiable, enabling price elasticity calculations.

**Reference:** Chiappori & Rochet (1987)

---

Strict Consistency (Acyclical P)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_strict_consistency(log)``

More lenient than GARP. Only checks cycles in strict preference :math:`P`:

.. math::

   \text{Acyclical P holds} \iff P^* \text{ has no cycles}

**Interpretation:** Approximately rational behavior. GARP may fail due to indifference, but strict preferences are consistent.

**Reference:** Dziewulski (2023)

---

Price Preferences (GAPP)
^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_price_preferences(log)``

Dual of GARP for price vectors. Define price preference:

.. math::

   p^s \, R_p \, p^t \iff p^s \cdot x^t \leq p^t \cdot x^t

**GAPP Condition:**

.. math::

   \text{GAPP holds} \iff \nexists \, s,t : \left( p^s \, R_p^* \, p^t \right) \land \left( p^t \, P_p \, p^s \right)

**Interpretation:** Consumer consistently prefers situations where desired bundles are cheaper.

**Reference:** Deb et al. (2022)

---

Efficiency Scores
-----------------

Integrity Score (Afriat Efficiency Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_integrity_score(log)``

**Definition:**

.. math::

   \text{AEI} = \sup \left\{ e \in [0,1] : \text{GARP holds with } e \cdot p^i \cdot x^i \geq p^i \cdot x^j \right\}

**Algorithm:** Binary search over :math:`e`. At efficiency :math:`e`, the modified revealed preference becomes:

.. math::

   x^i \, R_e \, x^j \iff e \cdot (p^i \cdot x^i) \geq p^i \cdot x^j

**Interpretation:**

- :math:`\text{AEI} = 1.0`: Perfectly consistent
- :math:`\text{AEI} = 0.8`: 20% of budget "wasted" on inconsistent choices
- :math:`\text{AEI} < 0.7`: Significant departures from rationality

**Reference:** Afriat (1972), Varian (1990)

---

Confusion Metric (Money Pump Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_confusion_metric(log)``

For a violation cycle :math:`k_1 \to k_2 \to \cdots \to k_m \to k_1`:

.. math::

   \text{MPI} = \frac{\sum_{i=1}^{m} \left( p^{k_i} \cdot x^{k_i} - p^{k_i} \cdot x^{k_{i+1}} \right)}{\sum_{i=1}^{m} p^{k_i} \cdot x^{k_i}}

**Interpretation:** Maximum fraction of spending extractable by cycling through preference contradictions.

- :math:`\text{MPI} = 0`: Unexploitable (consistent)
- :math:`\text{MPI} > 0.3`: Highly confused/exploitable

**Reference:** Chambers & Echenique (2016) Ch. 5

---

Outlier Fraction (Houtman-Maks Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_minimal_outlier_fraction(log)``

.. math::

   \text{HM} = \min \left\{ \frac{|S|}{T} : \text{removing observations } S \text{ makes data GARP-consistent} \right\}

**Reference:** Houtman & Maks (1985)

---

Per-Observation Efficiency (Varian's Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

---

Test Power (Bronars' Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_test_power(log)``

.. math::

   \text{Power} = \frac{\#\{\text{random behaviors violating GARP}\}}{\#\{\text{simulations}\}}

Random bundles are generated via symmetric Dirichlet distribution on budget hyperplanes:

.. math::

   \text{shares} \sim \text{Dirichlet}(1, 1, \ldots, 1), \quad x^t_j = \frac{\text{share}_j \cdot e_t}{p^t_j}

**Interpretation:**

- Power > 0.7: Passing GARP is statistically meaningful
- Power < 0.5: Even random behavior would pass

**Reference:** Bronars (1987)

---

Preference Structure
--------------------

Proportional Scaling (HARP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

---

Income Invariance (Quasilinearity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``test_income_invariance(log)``

Tests quasilinear utility :math:`U(x, m) = v(x) + m` via cyclic monotonicity.

**Condition:** For any cycle :math:`i_1 \to i_2 \to \cdots \to i_m \to i_1`:

.. math::

   \sum_{k=1}^{m} p^{i_k} \cdot (x^{i_{k+1}} - x^{i_k}) \geq 0

**Interpretation:** Choices depend only on relative prices, not income level.

**Reference:** Rochet (1987)

---

Feature Independence (Separability)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``test_feature_independence(log, group_a, group_b)``

Tests weak separability: :math:`U(x_A, x_B) = V(u_A(x_A), u_B(x_B))`

**Heuristic Test:**

1. Compute AEI within each group
2. Measure cross-group correlation
3. Separable if :math:`\text{AEI}_A > 0.9`, :math:`\text{AEI}_B > 0.9`, and cross-effect < 0.2

**Reference:** Chambers & Echenique (2016) Ch. 4, Theorem 4.4

---

Cross-Price Effects
^^^^^^^^^^^^^^^^^^^

**Function:** ``test_cross_price_effect(log, good_g, good_h)``

**Gross Substitutes:**

.. math::

   \Delta p_g > 0, \, \Delta x_g < 0 \implies \Delta x_h > 0

**Gross Complements:**

.. math::

   \Delta p_g > 0, \, \Delta x_g < 0 \implies \Delta x_h < 0

**Reference:** Hicks (1939)

---

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

---

References
----------

1. Afriat, S. N. (1967). The construction of utility functions from expenditure data. *International Economic Review*, 8(1), 67-77.

2. Bronars, S. G. (1987). The power of nonparametric tests of preference maximization. *Econometrica*, 55(3), 693-698.

3. Chambers, C. P., & Echenique, F. (2016). *Revealed Preference Theory*. Cambridge University Press.

4. Chiappori, P. A., & Rochet, J. C. (1987). Revealed preferences and differentiable demand. *Econometrica*, 55(3), 687-691.

5. Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2022). Revealed price preference: Theory and empirical analysis. *Review of Economic Studies*, forthcoming.

6. Dziewulski, P. (2023). Revealed preference and limited consideration. *American Economic Review*, forthcoming.

7. Hicks, J. R. (1939). *Value and Capital*. Oxford University Press.

8. Houtman, M., & Maks, J. A. H. (1985). Determining all maximal data subsets consistent with revealed preference. *Kwantitatieve Methoden*, 19, 89-104.

9. Rochet, J. C. (1987). A necessary and sufficient condition for rationalizability in a quasi-linear context. *Journal of Mathematical Economics*, 16(2), 191-200.

10. Samuelson, P. A. (1938). A note on the pure theory of consumer's behaviour. *Economica*, 5(17), 61-71.

11. Varian, H. R. (1982). The nonparametric approach to demand analysis. *Econometrica*, 50(4), 945-973.

12. Varian, H. R. (1983). Non-parametric tests of consumer behaviour. *Review of Economic Studies*, 50(1), 99-110.

13. Varian, H. R. (1990). Goodness-of-fit in optimizing models. *Journal of Econometrics*, 46(1-2), 125-140.
