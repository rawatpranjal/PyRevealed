Stochastic Choice
=================

Analyze probabilistic choice data with random utility models.
Based on Chambers & Echenique (2016) Chapter 13.

Random Utility Model
--------------------

**Function:** ``fit_random_utility_model(log)``

In a **Random Utility Model (RUM)**, utility has a deterministic and stochastic component:

.. math::

   U_i = V_i + \varepsilon_i

where :math:`V_i` is the systematic utility and :math:`\varepsilon_i` is a random shock.

**Choice Probability:**

.. math::

   P(i | A) = \Pr\left[ V_i + \varepsilon_i > V_j + \varepsilon_j \, \forall j \in A \right]

Logit Model
-----------

**Function:** ``fit_luce_model(log)``

With Type I Extreme Value errors, choice probabilities follow the **logit** form:

.. math::

   P(i | A) = \frac{\exp(V_i)}{\sum_{j \in A} \exp(V_j)}

This is equivalent to Luce's choice axiom.

McFadden's Axioms
-----------------

**Function:** ``test_mcfadden_axioms(log)``

RUM-consistent choice must satisfy:

1. **Regularity:** :math:`P(i | A) \geq P(i | B)` if :math:`A \subseteq B` and :math:`i \in A`

2. **IIA (for logit):** :math:`\frac{P(i | A)}{P(j | A)} = \frac{P(i | B)}{P(j | B)}` if :math:`i, j \in A \cap B`

.. admonition:: McFadden's Theorem
   :class: important

   Choice probabilities are consistent with RUM if and only if they satisfy a set of
   linear inequalities (Block-Marschak conditions).

**Reference:** McFadden (1974), Block & Marschak (1960), Chambers & Echenique (2016) Ch. 13
