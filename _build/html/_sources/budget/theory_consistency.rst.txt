Axiomatic Consistency Tests
==========================

Every choice adds edges to a directed **observation graph** (nodes = shopping trips, edges = revealed preferences). The axioms below define what "acyclic" means for this graph — GARP (allowing indifference), SARP (strict), and WARP (pairwise only).

GARP (Generalized Axiom of Revealed Preference)
-----------------------------------------------

.. image:: ../_static/garp_violation.gif
   :width: 55%
   :align: center
   :alt: GARP violation detection — building preference graph then tracing violation cycle

**Reference Implementation:** ``validate_consistency(log)``

The Generalized Axiom of Revealed Preference (GARP) constitutes the central behavioral benchmark for optimizing agents.

**Formal Revealed Preference Relations:**

Consider an agent facing price-quantity observations :math:`\{(p^t, x^t)\}_{t=1}^T`. We define the weak revealed preference relation :math:`R` and the strict revealed preference relation :math:`P` as follows:

.. math::

   x^i \, R \, x^j \iff p^i \cdot x^i \geq p^i \cdot x^j

.. math::

   x^i \, P \, x^j \iff p^i \cdot x^i > p^i \cdot x^j

Let :math:`R^*` denote the transitive closure of the relation :math:`R`, representing the indirect revealed preference relation.

**The GARP Condition:**

.. math::

   \text{GARP is satisfied} \iff \nexists \, i,j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, P \, x^i \right)

.. admonition:: Afriat's Theorem (1967)
   :class: important

   For any given dataset :math:`\{(p^t, x^t)\}_{t=1}^T`, the following propositions are logically equivalent:

   1. The data satisfy the **Generalized Axiom of Revealed Preference (GARP)**.
   2. There exist positive scalars :math:`\{U_t\}` and :math:`\{\lambda_t > 0\}` that satisfy the **Afriat Inequalities**:
      :math:`U_s \leq U_t + \lambda_t \cdot p^t \cdot (x^s - x^t) \quad \forall s,t`
   3. The data are **rationalizable** by a continuous, monotonic, and concave utility function :math:`U(x)`.

Afriat's Theorem establishes that GARP is both necessary and sufficient for the existence of a well-behaved utility function that rationalizes the observed choices.

**References:** Afriat (1967), Varian (1982), Chambers & Echenique (2016).


WARP (Weak Axiom of Revealed Preference)
----------------------------------------

**Reference Implementation:** ``validate_consistency_weak(log)``

The Weak Axiom of Revealed Preference (WARP) is a foundational consistency condition that precludes direct pairwise contradictions.

**The WARP Condition:**

.. math::

   \text{WARP is satisfied} \iff \nexists \, i,j : \left( x^i \, R \, x^j \right) \land \left( x^j \, P \, x^i \right)

Unlike GARP, WARP evaluates only direct inconsistencies (cycles of length 2) and does not account for transitive contradictions across longer sequences of choices.

**Reference:** Samuelson (1938).


SARP (Strong Axiom of Revealed Preference)
------------------------------------------

**Reference Implementation:** ``validate_sarp(log)``

The Strong Axiom of Revealed Preference (SARP) extends consistency by imposing acyclicity on the indirect revealed preference relation, effectively prohibiting indifference cycles.

**The SARP Condition (Acyclicity):**

.. math::

   \text{SARP is satisfied} \iff \nexists \, i \neq j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, R^* \, x^i \right)

SARP is a more restrictive condition than GARP and is necessary for the identification of unique, single-valued demand functions.

**Reference:** Houthakker (1950), Chambers & Echenique (2016).


Smooth Preferences and Differentiable Utility
---------------------------------------------

**Reference Implementation:** ``validate_smooth_preferences(log)``

In econometric applications requiring differentiable utility specifications (e.g., for calculating price elasticities), the observed behavior must satisfy the following joint conditions:

1. **SARP Consistency**: Preclusion of all indifference cycles.
2. **Local Injectivity**: A unique mapping from prices to quantities such that:

.. math::

   p^t \neq p^s \implies x^t \neq x^s

**Reference:** Chiappori & Rochet (1987).


Acyclical Strict Preferences (Acyclical P)
------------------------------------------

**Reference Implementation:** ``validate_strict_consistency(log)``

This condition provides a more lenient consistency test by evaluating cycles exclusively within the strict revealed preference relation :math:`P`:

.. math::

   \text{Acyclical P holds} \iff P^* \text{ contains no cycles}

This criterion accounts for indifference by allowing cycles in the weak relation :math:`R`, provided no strict preference is violated.

**Reference:** Dziewulski (2023).


Generalized Axiom of Price Preferences (GAPP)
---------------------------------------------

**Reference Implementation:** ``validate_price_preferences(log)``

GAPP constitutes the dual of GARP in the space of price vectors. We define the price preference relation :math:`R_p`:

.. math::

   p^s \, R_p \, p^t \iff p^s \cdot x^t \leq p^t \cdot x^t

**The GAPP Condition:**

.. math::

   \text{GAPP is satisfied} \iff \nexists \, s,t : \left( p^s \, R_p^* \, p^t \right) \land \left( p^t \, P_p \, p^s \right)

This dual test evaluates whether an agent exhibits consistent preferences across different budget environments, independent of specific quantity bundles.

**Reference:** Deb et al. (2022).
