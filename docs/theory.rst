Theory
======

.. note::

   This page provides formal mathematical definitions for all methods in PyRevealed,
   based on *Revealed Preference Theory* by Chambers & Echenique (2016).

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


Maintained Assumptions
----------------------

The following assumptions are **necessary** for revealed preference analysis to be meaningful.
If these are violated, GARP failures may reflect model misspecification rather than behavioral inconsistency.

.. list-table::
   :header-rows: 1
   :widths: 5 25 70

   * -
     - Assumption
     - Implication if Violated
   * - **A1**
     - **Stable Preferences** - Consumer has a fixed utility function :math:`U(x)` across all observations
     - Legitimate preference changes (e.g., developing a taste for coffee) appear as GARP violations
   * - **A2**
     - **Utility Maximization** - Consumer chooses :math:`\arg\max_x U(x)` subject to budget
     - Satisficing, habit formation, inattention, and heuristic decision-making generate violations
   * - **A3**
     - **Local Non-Satiation** - More is always weakly preferred; consumer spends entire budget
     - Free disposal allowed, but discarding goods or saving violates the model
   * - **A4**
     - **Single Decision-Maker** - Observed choices reflect one agent's preferences
     - Household data with multiple members, or accounts with gift purchases, violate this
   * - **A5**
     - **Complete Observation** - We observe the entire consumption bundle and prices faced
     - If we only see partial consumption (e.g., Amazon but not groceries), GARP may fail spuriously


Axiom Hierarchy
---------------

.. admonition:: Relationship Between Axioms

   **SARP** :math:`\Rightarrow` **GARP** :math:`\Rightarrow` **WARP**

   - **WARP** rules out direct contradictions (length-2 cycles)
   - **GARP** rules out transitive contradictions (any cycle with a strict edge)
   - **SARP** rules out all indifference cycles (strongest condition)

   Most empirical work uses GARP as it corresponds exactly to utility maximization via Afriat's Theorem.


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

.. admonition:: Afriat's Theorem (1967)
   :class: important

   The following are **equivalent**:

   1. The data :math:`\{(p^t, x^t)\}_{t=1}^T` satisfy GARP
   2. There exist utility values :math:`\{U_t\}` and multipliers :math:`\{\lambda_t > 0\}` satisfying:
      :math:`U_s \leq U_t + \lambda_t \cdot p^t \cdot (x^s - x^t) \quad \forall s,t`
   3. There exists a **continuous, monotonic, concave** utility function :math:`U(x)` that rationalizes the data

   This is the foundational result: GARP is both necessary and sufficient for rationalizability.

**Reference:** Afriat (1967), Varian (1982), Chambers & Echenique (2016) Ch. 3


WARP (Weak Axiom of Revealed Preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_consistency_weak(log)``

**WARP Condition:**

.. math::

   \text{WARP holds} \iff \nexists \, i,j : \left( x^i \, R \, x^j \right) \land \left( x^j \, P \, x^i \right)

Unlike GARP, WARP only checks direct (length-2) violations without transitivity.

**Reference:** Samuelson (1938), Chambers & Echenique (2016) Ch. 2


SARP (Strict Axiom of Revealed Preference)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_sarp(log)``

**SARP Condition (Antisymmetry):**

.. math::

   \text{SARP holds} \iff \nexists \, i \neq j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, R^* \, x^i \right)

SARP prohibits indifference cycles. Stronger than GARP.

**Reference:** Chambers & Echenique (2016) Ch. 2


Smooth Preferences (Differentiable Utility)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_smooth_preferences(log)``

Two conditions must hold:

1. **SARP:** No indifference cycles
2. **Price-Quantity Uniqueness:**

.. math::

   p^t \neq p^s \implies x^t \neq x^s

**Interpretation:** Demand function is well-defined and differentiable, enabling price elasticity calculations.

**Reference:** Chiappori & Rochet (1987)


Strict Consistency (Acyclical P)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_strict_consistency(log)``

More lenient than GARP. Only checks cycles in strict preference :math:`P`:

.. math::

   \text{Acyclical P holds} \iff P^* \text{ has no cycles}

**Interpretation:** Approximately consistent behavior. GARP may fail due to indifference, but strict preferences are consistent.

**Reference:** Dziewulski (2023)


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

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - AEI
     - Interpretation
   * - 1.0
     - Perfectly consistent - all choices consistent with utility maximization
   * - 0.95+
     - Minor deviations - Varian's threshold for approximate consistency
   * - 0.85–0.95
     - Moderate deviations - some inconsistent choices
   * - < 0.70
     - Substantial departures from consistency

.. note::

   **Benchmark:** CKMS (2014) found mean AEI = 0.881 in controlled lab experiments.
   E-commerce data typically shows lower values due to measurement noise.

**Reference:** Afriat (1972), Varian (1990)


Confusion Metric (Money Pump Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   * - 0.01–0.10
     - Minor cycles - small inconsistencies
   * - 0.10–0.30
     - Moderate cycles - noticeable inconsistencies
   * - > 0.30
     - Severe cycles - substantial inconsistencies

**Reference:** Chambers & Echenique (2016) Ch. 5


Outlier Fraction (Houtman-Maks Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_minimal_outlier_fraction(log)``

.. math::

   \text{HM} = \min \left\{ \frac{|S|}{T} : \text{removing observations } S \text{ makes data GARP-consistent} \right\}

**Reference:** Houtman & Maks (1985)


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


Test Power (Bronars' Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   * - 0.70–0.90
     - Good - meaningful test of consistency
   * - 0.50–0.70
     - Moderate - some discrimination but inconclusive
   * - < 0.50
     - Weak - even random behavior often passes; test lacks power

.. warning::

   Low power indicates budget sets don't overlap much. A high AEI with low power
   may just mean the data couldn't detect inconsistency, not that the consumer is consistent.

**Reference:** Bronars (1987)


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


Income Invariance (Quasilinearity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``test_income_invariance(log)``

Tests quasilinear utility :math:`U(x, m) = v(x) + m` via cyclic monotonicity.

**Condition:** For any cycle :math:`i_1 \to i_2 \to \cdots \to i_m \to i_1`:

.. math::

   \sum_{k=1}^{m} p^{i_k} \cdot (x^{i_{k+1}} - x^{i_k}) \geq 0

**Interpretation:** Choices depend only on relative prices, not income level.

**Reference:** Rochet (1987)


Feature Independence (Separability)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``test_feature_independence(log, group_a, group_b)``

Tests weak separability: :math:`U(x_A, x_B) = V(u_A(x_A), u_B(x_B))`

**Heuristic Test:**

1. Compute AEI within each group
2. Measure cross-group correlation
3. Separable if :math:`\text{AEI}_A > 0.9`, :math:`\text{AEI}_B > 0.9`, and cross-effect < 0.2

**Reference:** Chambers & Echenique (2016) Ch. 4, Theorem 4.4


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


References
----------

1. Afriat, S. N. (1967). The construction of utility functions from expenditure data. *International Economic Review*, 8(1), 67-77.

2. Bronars, S. G. (1987). The power of nonparametric tests of preference maximization. *Econometrica*, 55(3), 693-698.

3. Chambers, C. P., & Echenique, F. (2016). *Revealed Preference Theory*. Cambridge University Press.

4. Choi, S., Kariv, S., Müller, W., & Silverman, D. (2014). Who is (more) rational? *American Economic Review*, 104(6), 1518-1550.

5. Chiappori, P. A., & Rochet, J. C. (1987). Revealed preferences and differentiable demand. *Econometrica*, 55(3), 687-691.

6. Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2022). Revealed price preference: Theory and empirical analysis. *Review of Economic Studies*, forthcoming.

7. Dziewulski, P. (2023). Revealed preference and limited consideration. *American Economic Review*, forthcoming.

8. Hicks, J. R. (1939). *Value and Capital*. Oxford University Press.

9. Houtman, M., & Maks, J. A. H. (1985). Determining all maximal data subsets consistent with revealed preference. *Kwantitatieve Methoden*, 19, 89-104.

10. Rochet, J. C. (1987). A necessary and sufficient condition for rationalizability in a quasi-linear context. *Journal of Mathematical Economics*, 16(2), 191-200.

11. Samuelson, P. A. (1938). A note on the pure theory of consumer's behaviour. *Economica*, 5(17), 61-71.

12. Varian, H. R. (1982). The nonparametric approach to demand analysis. *Econometrica*, 50(4), 945-973.

13. Varian, H. R. (1983). Non-parametric tests of consumer behaviour. *Review of Economic Studies*, 50(1), 99-110.

14. Varian, H. R. (1990). Goodness-of-fit in optimizing models. *Journal of Econometrics*, 46(1-2), 125-140.
