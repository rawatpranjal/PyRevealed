Consistency Tests
=================

This section covers the formal definitions of consistency axioms: GARP, WARP, SARP, and related tests.

GARP (Generalized Axiom of Revealed Preference)
-----------------------------------------------

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
----------------------------------------

**Function:** ``validate_consistency_weak(log)``

**WARP Condition:**

.. math::

   \text{WARP holds} \iff \nexists \, i,j : \left( x^i \, R \, x^j \right) \land \left( x^j \, P \, x^i \right)

Unlike GARP, WARP only checks direct (length-2) violations without transitivity.

**Reference:** Samuelson (1938), Chambers & Echenique (2016) Ch. 2


SARP (Strict Axiom of Revealed Preference)
------------------------------------------

**Function:** ``validate_sarp(log)``

**SARP Condition (Antisymmetry):**

.. math::

   \text{SARP holds} \iff \nexists \, i \neq j : \left( x^i \, R^* \, x^j \right) \land \left( x^j \, R^* \, x^i \right)

SARP prohibits indifference cycles. Stronger than GARP.

**Reference:** Chambers & Echenique (2016) Ch. 2


Smooth Preferences (Differentiable Utility)
-------------------------------------------

**Function:** ``validate_smooth_preferences(log)``

Two conditions must hold:

1. **SARP:** No indifference cycles
2. **Price-Quantity Uniqueness:**

.. math::

   p^t \neq p^s \implies x^t \neq x^s

**Interpretation:** Demand function is well-defined and differentiable, enabling price elasticity calculations.

**Reference:** Chiappori & Rochet (1987)


Strict Consistency (Acyclical P)
--------------------------------

**Function:** ``validate_strict_consistency(log)``

More lenient than GARP. Only checks cycles in strict preference :math:`P`:

.. math::

   \text{Acyclical P holds} \iff P^* \text{ has no cycles}

**Interpretation:** Approximately consistent behavior. GARP may fail due to indifference, but strict preferences are consistent.

**Reference:** Dziewulski (2023)


Price Preferences (GAPP)
------------------------

**Function:** ``validate_price_preferences(log)``

Dual of GARP for price vectors. Define price preference:

.. math::

   p^s \, R_p \, p^t \iff p^s \cdot x^t \leq p^t \cdot x^t

**GAPP Condition:**

.. math::

   \text{GAPP holds} \iff \nexists \, s,t : \left( p^s \, R_p^* \, p^t \right) \land \left( p^t \, P_p \, p^s \right)

**Interpretation:** Consumer consistently prefers situations where desired bundles are cheaper.

**Reference:** Deb et al. (2022)
