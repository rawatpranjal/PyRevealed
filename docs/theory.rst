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


Abstract Choice Theory (Menu-Based)
-----------------------------------

Abstract choice theory analyzes preferences from discrete choices **without prices**.
Instead of observing (price, quantity) pairs, we observe (menu, choice) pairs where
each menu is a finite set of alternatives and the choice is one element from that menu.

This framework applies to:

- **Surveys**: "Which of these do you prefer?"
- **Recommendations**: "User clicked item X from shown set Y"
- **Voting**: "Candidate selected from ballot options"
- **A/B tests**: "Variant chosen from available options"

Notation (Menu-Based)
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80

   * - :math:`B_t \subseteq X`
     - Menu at observation :math:`t` (finite set of alternatives)
   * - :math:`c(B_t) \in B_t`
     - Chosen item from menu :math:`B_t`
   * - :math:`x \, R \, y`
     - :math:`x` is revealed preferred to :math:`y` (x chosen when y available)
   * - :math:`R^*`
     - Transitive closure of :math:`R`
   * - :math:`T`
     - Number of observations

Revealed Preference Relation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For menu-based choices, the revealed preference relation is defined as:

.. math::

   x \, R \, y \iff \exists \, t : c(B_t) = x \text{ and } y \in B_t

In words: :math:`x` is revealed preferred to :math:`y` if :math:`x` was chosen from
a menu containing :math:`y`.


WARP for Menus
^^^^^^^^^^^^^^

**Function:** ``validate_menu_warp(log)``

WARP (Weak Axiom of Revealed Preference) prohibits **direct contradictions**:

.. math::

   \text{WARP holds} \iff \nexists \, x, y : (x \, R \, y) \land (y \, R \, x)

If :math:`x` was chosen over :math:`y` in one menu, then :math:`y` cannot be
chosen over :math:`x` in another menu.

**Example violation:** User chose Pizza over Burger, then later chose Burger over Pizza.


SARP for Menus
^^^^^^^^^^^^^^

**Function:** ``validate_menu_sarp(log)``

SARP (Strong Axiom of Revealed Preference) prohibits **cycles of any length**:

.. math::

   \text{SARP holds} \iff R^* \text{ is acyclic}

Equivalently:

.. math::

   \text{SARP holds} \iff \nexists \, x_1, \ldots, x_m : x_1 \, R \, x_2 \, R \, \cdots \, R \, x_m \, R \, x_1

**Example violation:** Pizza > Burger, Burger > Salad, Salad > Pizza (3-cycle)


Congruence (Full Rationalizability)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_menu_consistency(log)``

The Congruence axiom (Richter's condition) requires:

1. **SARP**: No preference cycles
2. **Maximality**: The chosen item is maximal under :math:`R^*` within the menu

.. admonition:: Richter's Theorem (1966)
   :class: important

   A choice function :math:`c` is rationalizable by a complete, transitive preference
   ordering **if and only if** it satisfies Congruence.


Houtman-Maks Efficiency (Menus)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_menu_efficiency(log)``

When SARP fails, measure how close the data is to consistency:

.. math::

   \text{HM} = 1 - \min \left\{ \frac{|S|}{T} : \text{removing observations } S \text{ yields SARP-consistent data} \right\}

**Interpretation:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Efficiency
     - Interpretation
   * - 1.0
     - All choices are consistent
   * - 0.9+
     - Minor inconsistencies (1-2 problematic choices)
   * - < 0.8
     - Substantial inconsistencies


Ordinal Utility Recovery
^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``fit_menu_preferences(log)``

If SARP holds, recover the preference ordering via topological sort of the revealed
preference graph. The result is a ranking where item :math:`i` ranked before item :math:`j`
means :math:`i \succ j` (i is preferred to j).

**Algorithm:**

1. Build revealed preference graph: edge :math:`x \to y` if :math:`x \, R \, y`
2. Compute transitive closure :math:`R^*`
3. Topological sort of :math:`R^*` gives preference order (most preferred first)

**Note:** If multiple orderings are compatible with the data, one consistent ordering
is returned.

**Reference:** Chambers & Echenique (2016) Ch. 1-2, Richter (1966)


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

11. Richter, M. K. (1966). Revealed preference theory. *Econometrica*, 34(3), 635-645.

12. Samuelson, P. A. (1938). A note on the pure theory of consumer's behaviour. *Economica*, 5(17), 61-71.

13. Varian, H. R. (1982). The nonparametric approach to demand analysis. *Econometrica*, 50(4), 945-973.

14. Varian, H. R. (1983). Non-parametric tests of consumer behaviour. *Review of Economic Studies*, 50(1), 99-110.

15. Varian, H. R. (1990). Goodness-of-fit in optimizing models. *Journal of Econometrics*, 46(1-2), 125-140.

16. Hurwicz, L., & Uzawa, H. (1971). On the integrability of demand functions. In *Preferences, Utility, and Demand* (pp. 114-148). Harcourt Brace Jovanovich.

17. McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In *Frontiers in Econometrics* (pp. 105-142). Academic Press.

18. Block, H. D., & Marschak, J. (1960). Random orderings and stochastic theories of responses. *Contributions to Probability and Statistics*, 2, 97-132.

19. Masatlioglu, Y., Nakajima, D., & Ozbay, E. Y. (2012). Revealed attention. *American Economic Review*, 102(5), 2183-2205.

20. Varian, H. R. (1984). The nonparametric approach to production analysis. *Econometrica*, 52(3), 579-597.
