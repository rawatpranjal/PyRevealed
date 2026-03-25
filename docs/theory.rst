Theory
======

.. note::

   Formal mathematical definitions for all methods in PyRevealed,
   based on *Revealed Preference Theory* by Chambers & Echenique (2016).
   For worked examples see :doc:`examples`.

.. contents:: Sections
   :local:
   :depth: 2


Foundations
----------

Notation
^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^

These are **necessary** for revealed preference analysis to be meaningful.
Violations cause GARP failures that reflect model misspecification, not behavioral inconsistency.

- **A1 -- Stable Preferences.** Consumer has a fixed :math:`U(x)` across observations.
- **A2 -- Utility Maximization.** Consumer chooses :math:`\arg\max_x U(x)` subject to budget.
- **A3 -- Local Non-Satiation.** More is weakly preferred; entire budget is spent.
- **A4 -- Single Decision-Maker.** Choices reflect one agent's preferences.
- **A5 -- Complete Observation.** We observe the full consumption bundle and prices faced.

Axiom Hierarchy
^^^^^^^^^^^^^^^

**SARP** :math:`\Rightarrow` **GARP** :math:`\Rightarrow` **WARP**

- **WARP** rules out direct contradictions (length-2 cycles).
- **GARP** rules out transitive contradictions (any cycle with a strict edge).
- **SARP** rules out all indifference cycles (strongest condition).

Most empirical work uses GARP (corresponds exactly to utility maximization via Afriat's Theorem).
WARP serves as a quick sanity check. SARP is needed for smooth/differentiable utility functions.


Consistency Axioms
------------------

Revealed Preference Relations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define weak revealed preference :math:`R` and strict revealed preference :math:`P`:

.. math::

   x^i \, R \, x^j \iff p^i \cdot x^i \geq p^i \cdot x^j

.. math::

   x^i \, P \, x^j \iff p^i \cdot x^i > p^i \cdot x^j

Let :math:`R^*` denote the transitive closure of :math:`R` (computed via Floyd-Warshall).

GARP
^^^^

**Function:** ``validate_consistency(log)``

.. math::

   \text{GARP holds} \iff \nexists \, i,j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, P \, x^i \right)

.. admonition:: Afriat's Theorem (1967)
   :class: important

   The following are **equivalent**:

   1. The data :math:`\{(p^t, x^t)\}_{t=1}^T` satisfy GARP
   2. There exist :math:`\{U_t\}` and :math:`\{\lambda_t > 0\}` with
      :math:`U_s \leq U_t + \lambda_t \cdot p^t \cdot (x^s - x^t) \;\; \forall s,t`
   3. There exists a **continuous, monotonic, concave** :math:`U(x)` rationalizing the data

**Reference:** Afriat (1967), Varian (1982), Chambers & Echenique (2016) Ch. 3

WARP
^^^^

**Function:** ``validate_consistency_weak(log)``

.. math::

   \text{WARP holds} \iff \nexists \, i,j : \left( x^i \, R \, x^j \right) \land \left( x^j \, P \, x^i \right)

Checks direct (length-2) violations only, without transitivity.

**Reference:** Samuelson (1938)

SARP
^^^^

**Function:** ``validate_sarp(log)``

.. math::

   \text{SARP holds} \iff \nexists \, i \neq j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, R^* \, x^i \right)

Prohibits indifference cycles. Stronger than GARP.

**Reference:** Chambers & Echenique (2016) Ch. 2

Smooth Preferences
^^^^^^^^^^^^^^^^^^

**Function:** ``validate_smooth_preferences(log)``

Requires SARP and price-quantity uniqueness: :math:`p^t \neq p^s \implies x^t \neq x^s`.
Implies a well-defined, differentiable demand function. Reference: Chiappori & Rochet (1987).

Strict Consistency (Acyclical P)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_strict_consistency(log)``

.. math::

   \text{Acyclical P holds} \iff P^* \text{ has no cycles}

More lenient than GARP -- only checks cycles in strict preference. Reference: Dziewulski (2023).

GAPP (Price Preferences)
^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_price_preferences(log)``

Dual of GARP for price vectors:

.. math::

   p^s \, R_p \, p^t \iff p^s \cdot x^t \leq p^t \cdot x^t

.. math::

   \text{GAPP holds} \iff \nexists \, s,t : \left( p^s \, R_p^* \, p^t \right) \land \left( p^t \, P_p \, p^s \right)

**Reference:** Deb et al. (2022)


Efficiency Measures
-------------------

When GARP fails, these metrics quantify the severity of violations.

Integrity Score (Afriat Efficiency Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_integrity_score(log)``

.. math::

   \text{AEI} = \sup \left\{ e \in [0,1] : \text{GARP holds with } e \cdot p^i \cdot x^i \geq p^i \cdot x^j \right\}

Found via binary search. **Thresholds:** 1.0 = perfect; 0.95+ = minor deviations (Varian);
0.85--0.95 = moderate; < 0.70 = substantial. Benchmark: CKMS (2014) lab mean = 0.881.

**Reference:** Afriat (1972), Varian (1990)

Confusion Metric (Money Pump Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_confusion_metric(log)``

For a violation cycle :math:`k_1 \to k_2 \to \cdots \to k_m \to k_1`:

.. math::

   \text{MPI} = \frac{\sum_{i=1}^{m} \left( p^{k_i} \cdot x^{k_i} - p^{k_i} \cdot x^{k_{i+1}} \right)}{\sum_{i=1}^{m} p^{k_i} \cdot x^{k_i}}

Maximum fraction of spending extractable via preference cycles.
**Thresholds:** 0 = consistent; 0.01--0.10 = minor; 0.10--0.30 = moderate; > 0.30 = severe.

**Reference:** Echenique, Lee & Shum (2011)

Outlier Fraction (Houtman-Maks Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_minimal_outlier_fraction(log)``

.. math::

   \text{HM} = \min \left\{ \frac{|S|}{T} : \text{removing observations } S \text{ makes data GARP-consistent} \right\}

**Reference:** Houtman & Maks (1985)

Per-Observation Efficiency (Varian's Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_granular_integrity(log)``

.. math::

   \min \sum_{i=1}^{T} (1 - e_i) \quad \text{s.t.} \quad e_i \cdot (p^i \cdot x^i) \geq p^i \cdot x^j \;\; \forall \, (i,j) : x^i \, R^* \, x^j, \quad 0 \leq e_i \leq 1

Identifies which specific observations are problematic. Reference: Varian (1990).

Test Power (Bronars' Index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_test_power(log)``

.. math::

   \text{Power} = \frac{\#\{\text{random behaviors violating GARP}\}}{\#\{\text{simulations}\}}

Random bundles generated via symmetric Dirichlet on budget hyperplanes.
**Thresholds:** > 0.90 = excellent; 0.70--0.90 = good; 0.50--0.70 = moderate; < 0.50 = weak.

.. warning::

   Low power means budget sets don't overlap much. High AEI with low power
   may reflect insufficient data, not genuine consistency.

**Reference:** Bronars (1987)

Preference Structure
^^^^^^^^^^^^^^^^^^^^

**HARP (Homothetic):** ``validate_proportional_scaling(log)`` --
Define :math:`r_{ij} = p^i \cdot x^i / p^i \cdot x^j`. HARP holds iff no cycle has
:math:`\prod r_{i_k, i_{k+1}} > 1`. Reference: Varian (1983).

**Quasilinear:** ``test_income_invariance(log)`` --
For any cycle: :math:`\sum_k p^{i_k} \cdot (x^{i_{k+1}} - x^{i_k}) \geq 0`.
Reference: Rochet (1987).

**Separability:** ``test_feature_independence(log, group_a, group_b)`` --
Tests :math:`U(x_A, x_B) = V(u_A(x_A), u_B(x_B))` via within-group AEI and cross-group correlation.

**Utility Recovery:** ``fit_latent_values(log)`` --
If GARP holds, solve for :math:`U_k, \lambda_k > 0` satisfying
:math:`U_k \leq U_l + \lambda_l \cdot p^l \cdot (x^k - x^l) \;\; \forall k,l`.
Reference: Afriat (1967).


Menu-Based Choice
-----------------

Abstract choice theory analyzes preferences from discrete choices **without prices**.
We observe (menu, choice) pairs where each menu is a finite set of alternatives.

Notation
^^^^^^^^

.. list-table::
   :widths: 20 80

   * - :math:`B_t \subseteq X`
     - Menu at observation :math:`t` (finite set of alternatives)
   * - :math:`c(B_t) \in B_t`
     - Chosen item from menu :math:`B_t`
   * - :math:`x \, R \, y`
     - :math:`x` revealed preferred to :math:`y` (:math:`x` chosen when :math:`y` available)
   * - :math:`R^*`
     - Transitive closure of :math:`R`

Revealed preference: :math:`x \, R \, y \iff \exists \, t : c(B_t) = x \text{ and } y \in B_t`.

WARP for Menus
^^^^^^^^^^^^^^

**Function:** ``validate_menu_warp(log)``

.. math::

   \text{WARP holds} \iff \nexists \, x, y : (x \, R \, y) \land (y \, R \, x)

No direct preference reversals.

SARP for Menus
^^^^^^^^^^^^^^

**Function:** ``validate_menu_sarp(log)``

.. math::

   \text{SARP holds} \iff R^* \text{ is acyclic}

No preference cycles of any length.

Congruence (Full Rationalizability)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``validate_menu_consistency(log)``

Requires SARP plus maximality (chosen item is maximal under :math:`R^*` within the menu).

.. admonition:: Richter's Theorem (1966)
   :class: important

   A choice function :math:`c` is rationalizable by a complete, transitive preference
   ordering **if and only if** it satisfies Congruence.

Houtman-Maks Efficiency (Menus)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``compute_menu_efficiency(log)``

.. math::

   \text{HM} = 1 - \min \left\{ \frac{|S|}{T} : \text{removing } S \text{ yields SARP-consistent data} \right\}

**Thresholds:** 1.0 = all consistent; 0.9+ = minor; < 0.8 = substantial inconsistencies.

Ordinal Utility Recovery
^^^^^^^^^^^^^^^^^^^^^^^^

**Function:** ``fit_menu_preferences(log)``

If SARP holds, recover the preference ordering via topological sort of :math:`R^*`.
One consistent ordering is returned.

**Reference:** Chambers & Echenique (2016) Ch. 1--2, Richter (1966)


Extensions
----------

Integrability and Welfare
^^^^^^^^^^^^^^^^^^^^^^^^^

The integrability problem asks whether observed demand derives from utility maximization
via the Slutsky matrix :math:`S_{ij} = \partial h_i / \partial p_j`. Integrability requires
symmetry (:math:`S_{ij} = S_{ji}`) and negative semi-definiteness (:math:`v^\top S v \leq 0`)
-- the Hurwicz-Uzawa characterization (Ch. 6).

Welfare changes are measured by **compensating variation** :math:`CV = e(p^1, u^0) - e(p^0, u^0)`
and **equivalent variation** :math:`EV = e(p^1, u^1) - e(p^0, u^1)`, where :math:`e(p,u)` is
the expenditure function. For infinitesimal changes :math:`CV \approx EV`; for discrete changes
they differ unless preferences are quasilinear (Ch. 7).

**Functions:** ``compute_slutsky_matrix``, ``test_integrability``,
``compute_compensating_variation``, ``compute_equivalent_variation``

Additive Separability and Compensated Demand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Preferences are **additively separable** if :math:`U(x) = \sum_i u_i(x_i)`, implying
no cross-price effects. The testable restriction: cross-price effects are entirely income
effects, :math:`\partial x_i / \partial p_j = -x_j \cdot \partial x_i / \partial m` for
:math:`i \neq j` (Ch. 9). The **Slutsky decomposition** splits total price effects into
substitution and income components. **Hicksian demand**
:math:`h(p,u) = \arg\min_x \{p \cdot x : U(x) \geq u\}` satisfies Shephard's lemma and
the compensated law of demand (Ch. 10).

**Functions:** ``test_additive_separability``, ``decompose_price_effects``,
``compute_hicksian_demand``

Spatial Preferences
^^^^^^^^^^^^^^^^^^^

A decision-maker has **metric preferences** if there exists an ideal point :math:`z^*` with
:math:`x \succ y \iff d(x, z^*) < d(y, z^*)`. Supported metrics: Euclidean, Manhattan,
Chebyshev, Minkowski. The best-fitting metric minimizes violations across families (Ch. 11).

**Functions:** ``find_ideal_point_general``, ``determine_best_metric``

Stochastic Choice
^^^^^^^^^^^^^^^^^

In a **Random Utility Model (RUM)**, :math:`U_i = V_i + \varepsilon_i`, yielding
:math:`P(i|A) = \Pr[V_i + \varepsilon_i > V_j + \varepsilon_j \;\forall j \in A]`.
With Type I Extreme Value errors this gives the **logit model**:
:math:`P(i|A) = \exp(V_i) / \sum_j \exp(V_j)`. RUM-consistent choice satisfies
**regularity** (:math:`P(i|A) \geq P(i|B)` when :math:`A \subseteq B`) and, for logit,
**IIA**. McFadden's theorem: RUM-consistency :math:`\iff` Block-Marschak conditions (Ch. 13).

**Functions:** ``fit_random_utility_model``, ``fit_luce_model``, ``test_mcfadden_axioms``

Limited Attention
^^^^^^^^^^^^^^^^^

The decision-maker maximizes utility over a **consideration set**
:math:`\Gamma(A) \subseteq A`: :math:`c(A) = \arg\max_{x \in \Gamma(A)} u(x)`.
An **attention filter** satisfies
:math:`x \notin \Gamma(A) \implies \Gamma(A \setminus \{x\}) = \Gamma(A)`.
**WARP(LA)** (Masatlioglu et al., 2012) defines :math:`x \, P \, y` when removing :math:`y`
changes the choice away from :math:`x`; acyclicity of :math:`P` characterizes CLA.
The **Random Attention Model** (Cattaneo et al., 2020) extends this to stochastic choice
with item-level attention probabilities :math:`\mu_i` (Ch. 14).

**Functions:** ``test_attention_rationality``, ``test_warp_la``,
``recover_preference_with_attention``, ``estimate_consideration_sets``,
``fit_random_attention_model``, ``compute_attention_bounds``

Production Theory
^^^^^^^^^^^^^^^^^

Revealed preference extends to firm behavior. **Profit GARP** checks for arbitrage cycles:
if period-:math:`s` technology could have earned more at period-:math:`t` prices, the reverse
should not also hold. **Cost minimization** requires the input bundle is cheapest for the
observed output. **Returns to scale** (:math:`\partial \log y / \partial \log x`) are
estimated from production data (Ch. 15).

**Functions:** ``test_profit_maximization``, ``check_cost_minimization``,
``estimate_returns_to_scale``


References
----------

- Afriat, S. N. (1967). The construction of utility functions from expenditure data.
  *International Economic Review*.
- Block, H. D., & Marschak, J. (1960). Random orderings and stochastic theories of responses.
  *Contributions to Probability and Statistics*.
- Bronars, S. G. (1987). The power of nonparametric tests of preference maximization.
  *Econometrica*.
- Cattaneo, M. D., Ma, X., Masatlioglu, Y., & Suleymanov, E. (2020). A Random Attention Model.
  *Journal of Political Economy*.
- Chambers, C. P., & Echenique, F. (2016). *Revealed Preference Theory*. Cambridge University Press.
- Echenique, F., Lee, S., & Shum, M. (2011). The Money Pump as a measure of revealed
  preference violations. *Journal of Political Economy*.
- Houtman, M., & Maks, J. A. H. (1985). Determining all maximal data subsets consistent
  with revealed preference. *Kwantitatieve Methoden*.
- Masatlioglu, Y., Nakajima, D., & Ozbay, E. Y. (2012). Revealed attention.
  *American Economic Review*.
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
  *Frontiers in Econometrics*.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. *Econometrica*.
- Varian, H. R. (1990). Goodness-of-fit in optimizing models. *Journal of Econometrics*.
