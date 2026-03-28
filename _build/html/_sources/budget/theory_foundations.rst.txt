Theoretical Foundations
=======================

This chapter delineates the formal notation, foundational assumptions, and axiomatic hierarchy that constitute the theoretical core of revealed preference analysis.

Formal Notation
---------------

The analysis of choice behavior in PrefGraph is based on the following mathematical conventions:

.. list-table::
   :widths: 20 80

   * - :math:`p^t \in \mathbb{R}^n_+`
     - Price vector associated with observation :math:`t`.
   * - :math:`x^t \in \mathbb{R}^n_+`
     - Commodity bundle (quantity vector) selected at observation :math:`t`.
   * - :math:`e_t = p^t \cdot x^t`
     - Total expenditure at observation :math:`t`.
   * - :math:`E_{ij} = p^i \cdot x^j`
     - The hypothetical cost of bundle :math:`j` evaluated at the price vector of observation :math:`i`.
   * - :math:`T`
     - Total number of longitudinal observations for a given agent.
   * - :math:`n`
     - Dimensionality of the commodity space (number of distinct goods).


Maintained Assumptions
----------------------

The validity of revealed preference results is contingent upon several maintained assumptions regarding the underlying data-generating process. Violations of these assumptions may lead to spurious detections of behavioral inconsistency.

.. list-table::
   :header-rows: 1
   :widths: 5 25 70

   * -
     - Assumption
     - Implications of Violation
   * - **A1**
     - **Preference Stability** - The agent possesses a time-invariant utility function :math:`U(x)` across all observations.
     - Evolutionary changes in preferences (e.g., taste formation) may be incorrectly identified as axiomatic violations.
   * - **A2**
     - **Utility Maximization** - Observed choices represent the solution to :math:`\arg\max_x U(x)` subject to the budget constraint.
     - Decision heuristics, cognitive load, or satisficing behavior generate violations that reflect bounded rationality.
   * - **A3**
     - **Local Non-Satiation** - The agent strictly prefers more of at least one good; the entire budget is exhausted.
     - While free disposal is mathematically accommodated, systematic under-spending or unobserved saving violates the budget model.
   * - **A4**
     - **Unitary Decision-Maker** - Observed choices reflect the preferences of a single optimizing agent.
     - Aggregated household data or multi-user accounts may exhibit violations arising from collective choice dynamics.
   * - **A5**
     - **Information Completeness** - The analyst observes the exhaustive set of commodities and prices relevant to the agent's decision.
     - Partial observation (e.g., omitting essential categories) may result in an incomplete budget set, leading to false-positive violations.


Axiomatic Hierarchy
-------------------

The fundamental axioms of revealed preference exhibit a nested hierarchical structure, providing varying levels of stringency for behavioral analysis.

.. admonition:: Logical Relationship Between Axioms

   **SARP** :math:`\Rightarrow` **GARP** :math:`\Rightarrow` **WARP**

   - **WARP (Weak Axiom):** Precludes direct contradictions in pairwise choices (cycles of length 2).
   - **GARP (Generalized Axiom):** Precludes transitive contradictions across cycles of any length, provided at least one preference is strict.
   - **SARP (Strong Axiom):** Precludes all preference cycles, including those involving indifference; it is the most restrictive condition.

Empirical applications typically focus on **GARP**, as it provides the necessary and sufficient conditions for rationalizability by a continuous, monotonic, and concave utility function (Afriat, 1967).

.. note::

   **Axiomatic Selection Criteria:**

   - **WARP**: Employed as a computationally efficient preliminary filter for direct inconsistencies.
   - **GARP**: The standard benchmark for consumer rationality and utility maximization.
   - **SARP**: Required for applications necessitating unique demand systems or differentiable utility specifications.
