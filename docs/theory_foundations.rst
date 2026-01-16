Foundations
===========

This section covers the mathematical notation, maintained assumptions, and axiom hierarchy
that underpin all revealed preference analysis.

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

.. note::

   **When to use which axiom:**

   - **WARP**: Quick sanity check; if WARP fails, no need to test GARP
   - **GARP**: Standard test for rationality; corresponds to existence of a utility function
   - **SARP**: Strict test; required for smooth/differentiable utility functions
