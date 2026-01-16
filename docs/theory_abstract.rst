Abstract Choice Theory
======================

Abstract choice theory analyzes preferences from discrete choices **without prices**.
Instead of observing (price, quantity) pairs, we observe (menu, choice) pairs where
each menu is a finite set of alternatives and the choice is one element from that menu.

This framework applies to:

- **Surveys**: "Which of these do you prefer?"
- **Recommendations**: "User clicked item X from shown set Y"
- **Voting**: "Candidate selected from ballot options"
- **A/B tests**: "Variant chosen from available options"

Notation (Menu-Based)
---------------------

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
----------------------------

For menu-based choices, the revealed preference relation is defined as:

.. math::

   x \, R \, y \iff \exists \, t : c(B_t) = x \text{ and } y \in B_t

In words: :math:`x` is revealed preferred to :math:`y` if :math:`x` was chosen from
a menu containing :math:`y`.


WARP for Menus
--------------

**Function:** ``validate_menu_warp(log)``

WARP (Weak Axiom of Revealed Preference) prohibits **direct contradictions**:

.. math::

   \text{WARP holds} \iff \nexists \, x, y : (x \, R \, y) \land (y \, R \, x)

If :math:`x` was chosen over :math:`y` in one menu, then :math:`y` cannot be
chosen over :math:`x` in another menu.

**Example violation:** User chose Pizza over Burger, then later chose Burger over Pizza.


SARP for Menus
--------------

**Function:** ``validate_menu_sarp(log)``

SARP (Strong Axiom of Revealed Preference) prohibits **cycles of any length**:

.. math::

   \text{SARP holds} \iff R^* \text{ is acyclic}

Equivalently:

.. math::

   \text{SARP holds} \iff \nexists \, x_1, \ldots, x_m : x_1 \, R \, x_2 \, R \, \cdots \, R \, x_m \, R \, x_1

**Example violation:** Pizza > Burger, Burger > Salad, Salad > Pizza (3-cycle)


Congruence (Full Rationalizability)
-----------------------------------

**Function:** ``validate_menu_consistency(log)``

The Congruence axiom (Richter's condition) requires:

1. **SARP**: No preference cycles
2. **Maximality**: The chosen item is maximal under :math:`R^*` within the menu

.. admonition:: Richter's Theorem (1966)
   :class: important

   A choice function :math:`c` is rationalizable by a complete, transitive preference
   ordering **if and only if** it satisfies Congruence.


Houtman-Maks Efficiency (Menus)
-------------------------------

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
------------------------

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
