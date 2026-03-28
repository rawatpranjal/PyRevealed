Abstract Choice Theory and Menu-Based Analysis
===============================================

Abstract choice theory evaluates preference consistency from discrete choice observations in the absence of price vectors. In this framework, the analyst observes a sequence of menu-choice pairs, where each menu :math:`B_t` is a finite set of alternatives and the selection :math:`c(B_t)` is a single element from that menu.

This axiomatic framework is applicable to diverse empirical settings, including:

- **Survey Instruments:** Longitudinal evaluation of individual stated preferences.
- **Recommendation Systems:** Analysis of agent interactions within constrained digital interfaces.
- **Social Choice:** Examination of voting patterns across varied ballot options.
- **Experimental Design:** Discrete choice experiments and A/B testing protocols.

Formal Notation
---------------

The analysis of menu-based choices utilizes the following mathematical conventions:

.. list-table::
   :widths: 20 80

   * - :math:`B_t \subseteq X`
     - The menu (feasible set) available at observation :math:`t`.
   * - :math:`c(B_t) \in B_t`
     - The item selected by the agent from menu :math:`B_t`.
   * - :math:`x \, R \, y`
     - The weak revealed preference relation (where :math:`x` is chosen and :math:`y` was available).
   * - :math:`R^*`
     - The transitive closure of the revealed preference relation :math:`R`.
   * - :math:`T`
     - Total number of choice observations.

The Revealed Preference Relation
--------------------------------

For menu-based observations, the revealed preference relation :math:`R` is formally defined as:

.. math::

   x \, R \, y \iff \exists \, t : c(B_t) = x \text{ and } y \in B_t

This signifies that :math:`x` is revealed preferred to :math:`y` if :math:`x` was selected from a feasible set containing :math:`y`.


Weak Axiom of Revealed Preference (WARP)
----------------------------------------

**Reference Implementation:** ``validate_menu_warp(log)``

The Weak Axiom of Revealed Preference (WARP) in the context of discrete choice precludes direct pairwise contradictions in behavioral selections.

**The WARP Condition:**

.. math::

   \text{WARP is satisfied} \iff \nexists \, x, y : (x \, R \, y) \land (y \, R \, x)

If an agent selects :math:`x` when :math:`y` is available, WARP dictates that they cannot select :math:`y` in a subsequent menu where :math:`x` is also available.


Strong Axiom of Revealed Preference (SARP)
------------------------------------------

**Reference Implementation:** ``validate_menu_sarp(log)``

The Strong Axiom of Revealed Preference (SARP) extends consistency to transitivity, prohibiting preference cycles of any length within the discrete choice framework.

**The SARP Condition (Acyclicity):**

.. math::

   \text{SARP is satisfied} \iff R^* \text{ is acyclic}

Equivalently, SARP is satisfied if there exists no sequence :math:`x_1, \ldots, x_m` such that :math:`x_1 \, R \, x_2 \, R \, \cdots \, R \, x_m \, R \, x_1`.


Congruence and Full Rationalizability
-------------------------------------

**Reference Implementation:** ``validate_menu_consistency(log)``

The Congruence axiom, also known as Richter's condition, provides the necessary and sufficient criteria for the existence of a stable preference ordering.

.. admonition:: Richter's Theorem (1966)
   :class: important

   A discrete choice function :math:`c` is rationalizable by a complete and transitive preference ordering **if and only if** it satisfies the condition of Congruence.

Congruence requires both SARP (acyclicity) and the property that the selected item :math:`c(B_t)` is maximal under the indirect preference relation :math:`R^*` within the feasible set :math:`B_t`.


Houtman-Maks Efficiency for Discrete Choice
-------------------------------------------

**Reference Implementation:** ``compute_menu_efficiency(log)``

When behavioral data violate SARP, the Houtman-Maks Index quantifies the degree of approximate rationality by identifying the maximal subset of observations that satisfy axiomatic consistency.

**Formal Definition:**

.. math::

   \text{HM} = 1 - \min \left\{ \frac{|S|}{T} : \text{The subset } \{1, \ldots, T\} \setminus S \text{ satisfies SARP} \right\}

**Interpretation:**

An HM index of 1.0 indicates perfect transitivity, while lower values signify significant behavioral noise or model misspecification within the choice environment.


Ordinal Preference Recovery
---------------------------

**Reference Implementation:** ``fit_menu_preferences(log)``

If the agent's behavior satisfies SARP, the underlying ordinal preference ranking can be recovered via a topological sort of the item graph.

**Methodology:**

1. **Graph Construction:** Directed edges :math:`x \to y` are established for all observed relations :math:`x \, R \, y`.
2. **Transitive Extension:** The transitive closure :math:`R^*` is computed to identify all indirect preferences.
3. **Topological Ordering:** A linear ordering of alternatives is generated such that if :math:`x \, R^* \, y`, then :math:`x` is ranked before :math:`y`.

If multiple preference orderings are compatible with the observed data, PrefGraph returns one such consistent ranking.

**References:** Richter (1966), Chambers & Echenique (2016).


Item Graph Network Features
----------------------------

**Requires:** ``Engine(metrics=[..., "network"])``

The **item graph** :math:`G = (X, R)` has one node per alternative and a directed edge :math:`x \to y` whenever item :math:`x` was selected from a menu containing :math:`y`. In contrast to the budget observation graph (where nodes represent observations), nodes here represent items and edges encode direct pairwise preferences. These features characterize the distribution of revealed preferences across alternatives, independently of whether those preferences satisfy SARP or WARP.

**Graph Density** (``r_density``)

.. math::

   \rho = \frac{|\{(x,y) : x \, R \, y, \, x \neq y\}|}{|X|(|X|-1)}

The proportion of item pairs for which a direct revealed preference has been observed. Higher density indicates broader comparison coverage across the item space. Lower density indicates that many item pairs have not been directly compared in the observed menus.

**Preference Entropy** (``pref_entropy``)

Shannon entropy of the out-degree distribution, where :math:`d^+(x)` denotes the number of items over which :math:`x` has been selected:

.. math::

   H = -\sum_{x \in X} \frac{d^+(x)}{D} \log_2 \frac{d^+(x)}{D}, \quad D = \sum_{x} d^+(x)

Quantifies the dispersion of revealed preferences across items. High entropy indicates that selections are distributed across many alternatives. Low entropy indicates that a small number of items account for most selections. This measure is distinct from SARP consistency: an agent may exhibit high entropy while maintaining a fully acyclic preference ordering.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Interpretation
   * - High (near :math:`\log_2 |X|`)
     - Selections distributed uniformly across items. Consistent with variety-seeking behavior.
   * - Low (near 0)
     - Selections concentrated on a small subset of items. Consistent with stable item-level preferences.

**Choice Diversity** (``choice_diversity``)

.. math::

   \text{diversity} = \frac{|\{c(B_t) : t = 1, \ldots, T\}|}{T}

The ratio of distinct items selected to total observations. Values range from :math:`1/T` (a single item selected in every observation) to 1.0 (a distinct item selected in each observation). This measure captures the breadth of the agent's selection behavior, independent of menu composition or axiomatic consistency.
