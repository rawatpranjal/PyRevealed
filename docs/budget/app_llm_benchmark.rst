LLM Consistency Benchmarks
==========================

Do LLMs have stable action rankings, or does the ranking change when
different alternatives are shown? We build preference graphs from LLM
decisions and check for cycles.

.. image:: ../_static/robot_stock.svg
   :alt: Robot illustration
   :align: center
   :width: 320



**TL;DR.** GPT-4o-mini usually keeps a stable ranking of actions.
Between 74 to 92 percent of vignettes pass SARP at temperature 0, and
majority vote over 20 stochastic repetitions changes little, with
agreement around 96 to 98 percent. Where inconsistency appears, it is
systematic. It concentrates on adjacent severity pairs, shows compromise
and anchoring patterns, and persists under stochastic aggregation. Job
screening is the weakest case at 74 percent with many IIA violations.
Alert triage is the strongest at 92 percent with very few violations.
There is no single best prompt. Decision tree is perfect on procurement
but performs poorly on jobs.

.. image:: ../_static/app_llm_benchmark_summary.png
   :alt: LLM Benchmark Summary
   :align: center

.. code-block:: text

   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │ Vignettes  │───▶│ Query LLM  │───▶│ Build pref │───▶│ Test SARP  │
   │ (10 per    │    │ (15 menus  │    │ graph per  │    │ + IIA per  │
   │ scenario)  │    │ per vig.)  │    │ vignette   │    │ vignette   │
   └────────────┘    └────────────┘    └────────────┘    └────────────┘

Roadmap
-------

This page is a guided tour rather than a wall of numbers. We begin with
the :ref:`llm-setup`, then explain :ref:`llm-how-to-read` the figures and
tables. The results come in three passes. :ref:`llm-det-results` shows the
deterministic baseline. :ref:`llm-stoch-results` checks stability when we
sample. :ref:`llm-rum` tests stochastic rationality directly. We then walk
through :ref:`llm-patterns` that explain why flips happen, followed by
actionable :ref:`llm-findings`. The final sections show how to
:ref:`llm-reproduce` and include an :ref:`llm-appendix` with pipeline,
metrics, and limitations.

.. _llm-setup:

Setup
-----

A **scenario** is a real enterprise decision task such as support triage
or alert routing. Each scenario has five **actions** the LLM can choose
from. A **vignette** is a single concrete input such as a specific
support ticket, alert payload, or candidate resume that the LLM must
respond to. A **prompt** is the system prompt persona that frames how the
LLM should approach the decision. A **menu** is a subset of the five
actions shown to the LLM for a given vignette. We present all
C(5,2)=10 pairwise menus + 5 size-3 menus = 15 menus per vignette.

5 scenarios, 10 vignettes each, 5 prompts, 15 menus per vignette.
Model: gpt-4o-mini. Deterministic (temp=0) and stochastic (temp=0.7,
K=20 reps per menu).

.. list-table::
   :header-rows: 1
   :widths: 14 50 36

   * - Scenario
     - Actions
     - Prompts
   * - Support
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
     - **minimal** (bare instruction), **decision-tree** (if/then rules),
       **conservative** (prefer escalation), **aggressive** (prefer automation),
       **chain-of-thought** (numbered reasoning)
   * - Alert
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
     - same 5 prompts, adapted per scenario
   * - Content
     - approve, warning, hide, remove+strike, suspend+legal
     - same 5
   * - Jobs
     - reject, hold, phone screen, technical, fast-track
     - same 5
   * - Procurement
     - auto-approve, tag, request quotes, escalate, deny
     - same 5

.. _llm-how-to-read:

How to read the results
-----------------------

SARP is a deterministic consistency check. We build a preference graph by
adding an edge from the chosen item to every unchosen item in the same
menu. SARP passes when the transitive closure of this graph has no
cycles. That is equivalent to having a strict ranking that explains all
choices.

IIA is about menu independence. We compare pairwise menus with the
corresponding triples. If A beats B in the pair {A, B}, but adding C
shifts the choice to B in {A, B, C}, then the result depends on the menu
and independence is violated.

Stochastic results use K=20 samples at temperature 0.7 per menu and
report the majority choice. Agreement measures the percent of menus where
this majority matches the deterministic pick. Percent mixed is the share
of menus where the K responses do not all agree.

.. _llm-why-design:

Why this design
---------------

Holding the vignette constant and only changing the menu isolates menu
effects. Earlier designs that varied both content and menus at the same
time made it hard to tell whether a flip was caused by a different story
or a different set of options. This setup pins flips on the menu.

.. _llm-det-results:

Results 1: Deterministic (temp=0)
---------------------------------

We first measure consistency with no sampling. Each cell shows the
percent of vignette plus prompt pairs whose preference graph is acyclic.

.. list-table:: SARP pass rate by prompt
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - Mean
   * - Support
     - 90
     - 80
     - 100
     - 80
     - 90
     - 88
   * - Alert
     - 80
     - 100
     - 90
     - 100
     - 90
     - 92
   * - Content
     - 90
     - 80
     - 70
     - 90
     - 80
     - 82
   * - Jobs
     - 70
     - 60
     - 80
     - 80
     - 80
     - 74
   * - Procurement
     - 70
     - 100
     - 70
     - 90
     - 90
     - 84

This table compares prompts within each scenario. Higher values mean the
model maintains a single implied ranking across the 15 menus for that
vignette and prompt. Use it to pick a default prompt per scenario.

.. list-table:: SARP pass rate by vignette difficulty
   :header-rows: 1
   :widths: 18 16 16 16 16 16

   * -
     - Clear
     - Binary
     - Ambig.
     - Advers.
     - Mean
   * - Support
     - 87
     - 93
     - 90
     - 80
     - 88
   * - Alert
     - 93
     - 100
     - 90
     - 80
     - 92
   * - Content
     - 60
     - 80
     - 100
     - 100
     - 82
   * - Jobs
     - 87
     - 67
     - 70
     - 70
     - 74
   * - Procurement
     - 93
     - 73
     - 90
     - 80
     - 84

This table holds the prompt mix fixed and varies the vignette difficulty.
It shows which scenarios degrade when cases are clear versus ambiguous
or adversarial. Content is notably fragile on clear cases.

.. _llm-iia:

.. list-table:: IIA violations
   :header-rows: 1
   :widths: 20 15 15 20

   * -
     - Deterministic
     - Stochastic
     - Agreement
   * - Support
     - 3
     - 3
     - 95.8%
   * - Alert
     - 2
     - 3
     - 96.6%
   * - Content
     - 9
     - 12
     - 95.5%
   * - Jobs
     - 15
     - 14
     - 97.6%
   * - Procurement
     - 8
     - 6
     - 97.7%

*IIA violation = adding a third option flips the pairwise preference.
Stochastic = majority-vote from K=20 reps. Agreement = % of menus
where temp=0 and temp=0.7 majority match. Procurement stochastic
based on 82% of expected data (41/50 vignette-prompt combos).*

Read this table as a stress test for menu dependence. Higher IIA counts
mean more flips when a third option is added. High agreement means the
stochastic sample reinforces the deterministic choice rather than
contradicting it.

.. _llm-stoch-results:

Results 2: Stochastic (majority vote, temp=0.7)
-----------------------------------------------

We repeat each menu K times at temperature 0.7 and take the majority
choice per menu. These tables show how often acyclicity survives
sampling, and how often menus produce mixed responses across K reps.

.. list-table:: SARP pass rate (majority-vote)
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - Mean
   * - Support
     - 90
     - 80
     - 90
     - 90
     - 100
     - 90
   * - Alert
     - 80
     - 90
     - 90
     - 100
     - 90
     - 90
   * - Content
     - 90
     - 80
     - 60
     - 80
     - 70
     - 76
   * - Jobs
     - 80
     - 60
     - 80
     - 80
     - 90
     - 78
   * - Procurement
     - 78
     - 100
     - 75
     - 88
     - 75
     - 83

These rates come from majority vote at temperature 0.7. They measure how
often sampling preserves an acyclic preference order for a vignette and
prompt. Compare to deterministic rates to see which prompts are robust.

.. list-table:: % of menus with mixed responses
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
     - All
   * - Support
     - 8
     - 20
     - 5
     - 11
     - 11
     - 11
   * - Alert
     - 8
     - 11
     - 7
     - 9
     - 7
     - 8
   * - Content
     - 14
     - 7
     - 16
     - 5
     - 17
     - 12
   * - Jobs
     - 9
     - 10
     - 9
     - 5
     - 9
     - 8
   * - Procurement
     - 5
     - 8
     - 24
     - 16
     - 5
     - 12

Percent mixed captures instability within the K repetitions for the same
menu. High values point to prompts and scenarios where the model splits
between two nearby options rather than committing to one.

.. list-table:: SARP pass rate by tier (majority-vote)
   :header-rows: 1
   :widths: 18 16 16 16 16 16

   * -
     - Clear
     - Binary
     - Ambig.
     - Advers.
     - Mean
   * - Support
     - 93
     - 100
     - 80
     - 80
     - 90
   * - Alert
     - 87
     - 100
     - 90
     - 80
     - 90
   * - Content
     - 47
     - 87
     - 80
     - 100
     - 76
   * - Jobs
     - 87
     - 73
     - 80
     - 70
     - 78
   * - Procurement
     - 87
     - 67
     - 100
     - 100\*
     - 83

*\*Procurement adversarial based on 1 observed vignette.*

This tier view under stochastic sampling shows how clarity interacts with
menu effects. The contrast for content is informative. Clear cases pass
less often than ambiguous or adversarial ones because severity anchors
become more influential in clear settings.

.. _llm-rum:

Results 3: Stochastic Choice (RUM)
----------------------------------

We aggregate the K responses into choice frequencies and test random
utility consistency. This complements majority vote and reveals whether
the observed probabilities admit a rationalizing distribution over
rankings.

Majority vote summarizes what the model usually picks. To test stochastic
choice directly, we aggregate the K responses per menu into frequencies and
check Random Utility Model consistency, regularity, IIA, and transitivity.

.. list-table:: RUM pass rate by prompt (percent of vignette plus prompt pairs)
   :header-rows: 1
   :widths: 18 13 13 13 13 13

   * -
     - Min
     - DecTree
     - Conserv
     - Aggress
     - CoT
   * - Support
     - 50
     - 30
     - 70
     - 50
     - 70
   * - Alert
     - 70
     - 60
     - 90
     - 70
     - 80
   * - Content
     - 50
     - 60
     - 60
     - 80
     - 50
   * - Jobs
     - 70
     - 40
     - 60
     - 80
     - 60
   * - Procurement\*
     - 67
     - 88
     - 50
     - 38
     - 63

These rates show which prompts are stochastically rational in each scenario.
Decision tree is strongest on procurement under RUM while aggressive is often
best on jobs. CoT and conservative tend to stabilize support and alert.

.. list-table:: RUM pass rate by tier (percent)
   :header-rows: 1
   :widths: 18 16 16 16 16

   * -
     - Clear
     - Binary
     - Ambig.
     - Advers.
   * - Support
     - 53
     - 67
     - 40
     - 50
   * - Alert
     - 67
     - 93
     - 70
     - 60
   * - Content
     - 13
     - 80
     - 70
     - 90
   * - Jobs
     - 80
     - 60
     - 50
     - 50
   * - Procurement\*
     - 47
     - 47
     - 100
     - 100

RUM confirms the earlier pattern. Content is weakest on clear cases and
strongest on adversarial. Jobs and procurement improve on ambiguous and
adversarial tiers. Results with asterisks are based on partial procurement
stage2 coverage.

.. _llm-patterns:

Patterns
--------

IIA violations are not evenly distributed. Decision-tree is the IIA
hotspot on job screening (6 of 14 stochastic violations), while it
scores 100% SARP on procurement. Conservative leads on content
moderation (4 of 12). The same prompt can be the most and least
consistent depending on the decision domain.

.. list-table:: IIA violations by prompt (stochastic, top entries)
   :header-rows: 1
   :widths: 22 22 16

   * - Scenario
     - Prompt
     - IIA count
   * - Jobs
     - decision_tree
     - 6
   * - Content
     - conservative
     - 4
   * - Content
     - decision_tree
     - 3
   * - Content
     - aggressive
     - 2
   * - Jobs
     - minimal
     - 2
   * - Procurement
     - minimal
     - 2
   * - Procurement
     - conservative
     - 2

**Compromise effect (job screening).** In most job screening IIA cases
the mechanism is the same. Adding an extreme option shifts the choice
toward the middle option. For example, the model prefers
*phone_screen* over *hold_for_review* in a pair. When shown the triple
{auto_reject, hold, phone_screen}, the choice moves to *hold*. The
reverse also appears. Adding *fast_track* can move a choice from
*auto_reject* to *technical_interview*. This is the familiar
compromise pattern.

**Severity anchor (content moderation).** Adding a lenient option pushes
the choice toward a stricter action, and adding an extreme option pushes
the choice toward a moderate action. For example, the model prefers
*remove_and_strike* over *suspend_and_legal* in a pair. When shown the
triple {approve, remove, suspend}, *suspend* wins. In the opposite
direction, adding *suspend* can make *remove* preferable to *approve*.
The model anchors to the extremes of the menu.

**Parse failures as a policy floor.** In severe content cases, all-mild
menus sometimes return PARSE_FAIL. The model refuses to pick a mild
action and outputs a severe action instead, even though it is not in the
menu. This is not random error. It reveals a safety floor that the model
will not cross. SARP counts this as a violation, but it is better read
as a constraint revealed by the model.

.. _llm-findings:

Findings
--------

Job screening is the least consistent scenario. The deterministic SARP
pass rate is 74 percent and the stochastic rate is 78 percent. We count
15 deterministic and 14 stochastic IIA violations. In practice this means
that adding a third candidate often changes which of two candidates is
preferred, a hallmark of the compromise pattern.

Content moderation shows that clear cases are not always stable. The
clear tier passes only 47 percent of the time under stochastic sampling
while ambiguous and adversarial tiers are much higher. Menu dependent
severity judgments remain even after majority voting. We still see 12
stochastic IIA violations, which confirms that context effects persist
when we aggregate.

Prompt effects are real and specific to the domain. Decision tree is the
only prompt to reach 100 percent on a scenario and it does so on
procurement. The same prompt scores 60 percent on jobs. Conservative
steadies support at 90 percent but scores 60 percent on content.
There is no universal best prompt. Alert triage is the most consistent
scenario at 90 percent under stochastic SARP, likely because the action
set has a clear ordinal severity ladder. Procurement with the
conservative prompt has 24 percent mixed menus, the highest of any
scenario and prompt. Spending authority choices are more sensitive to
sampling temperature when the prompt emphasizes caution.

Finally, stochastic sampling barely changes the big picture. Between 96
and 98 percent of menus agree between temperature 0 and the majority vote
at temperature 0.7. Only 8 to 12 percent of menus produce mixed responses
across 20 repetitions. The inconsistency we see is structural rather than
noise.

.. _llm-reproduce:

Reproduce
---------

.. code-block:: bash

   pip install prefgraph openai
   export OPENAI_API_KEY=your_key
   cd examples

   # Deterministic (3,750 calls)
   python -m applications.llm_benchmark.v2.generate_vignettes --all
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
   python -m applications.llm_benchmark.v2.analyze --all

   # Stochastic (75,000 calls)
   python -m applications.llm_benchmark.v2.run_benchmark --all --stage 2 --k 20
   python -m applications.llm_benchmark.v2.analyze --all --stage 2

.. _llm-appendix:

Appendix
--------

Pipeline
~~~~~~~~

.. code-block:: text

   1. VIGNETTES: 10 per scenario, 4 tiers (clear, binary, ambiguous,
      adversarial). Generated by gpt-4o-mini.

   2. MENUS: C(5,2)=10 pairwise + 5 size-3 = 15 menus per vignette.
      Same input, different options shown.

   3. QUERY: gpt-4o-mini. temp=0 (1 response) or temp=0.7 (K=20).

   4. GRAPH: Each choice adds edges: chosen → each unchosen item.

   5. TEST: SARP (acyclic?). IIA (pairwise vs triple comparison).

Code
~~~~

.. code-block:: python

   from prefgraph import MenuChoiceLog
   from prefgraph.algorithms.abstract_choice import validate_menu_sarp

   log = MenuChoiceLog(
       menus=[frozenset(r["menu"]) for r in records],
       choices=[r["choice"] for r in records],
   )
   result = validate_menu_sarp(log)
   # result.is_consistent, result.violations

Metrics
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Metric
     - Range
     - Meaning
   * - SARP pass rate
     - 0--100%
     - % of vignettes with acyclic preference graph
   * - IIA violations
     - 0--n
     - Adding a third option flips a pairwise preference
   * - % mixed
     - 0--100%
     - % of menus with different choices across K reps

Limitations
~~~~~~~~~~~

No ground truth (consistency ≠ accuracy). Synthetic vignettes. Single
model family. Procurement stochastic based on 82% of expected data.
Stochastic analysis uses majority-vote aggregation over K=20 reps, not
a formal RUM LP test. For stochastic rationality testing (regularity,
Block-Marschak) see :doc:`../menu/theory_stochastic`.

Worked Example
~~~~~~~~~~~~~~

One row from the benchmark, showing every concept in action:

- **Scenario:** Procurement Approval
- **Vignette:** *"A department has submitted a procurement request for
  office supplies totaling $150, which is well within the approved budget
  of $500 for the month. All items requested are standard supplies that
  have been pre-approved in the procurement policy without the need for
  further review."* (tier: clear)
- **Prompt:** minimal --- *"Process procurement requests. Pick one action
  from the list."*
- **Actions:** auto-approve, tag, request quotes, escalate, deny
- **Menu:** {auto-approve, tag} (one of 15 menus shown for this vignette)
- **Choice:** auto-approve

This vignette is "clear" --- reasonable people would agree on
auto-approve. The model does too, consistently across all 15 menus
(SARP pass). Now contrast with an adversarial vignette from job
screening:

- **Scenario:** Job Application Screen
- **Vignette:** *"The candidate has 5 years of Python experience building
  data pipelines at a mid-size fintech. No distributed systems work, but
  led a Kafka migration that reduced processing latency by 40%. Bootcamp
  grad, no CS degree."* (tier: binary)
- **Prompt:** decision-tree --- explicit if/then rules for each action
- **Menu 1:** {hold, phone-screen} --- model picks **phone-screen**
- **Menu 2:** {auto-reject, hold, phone-screen} --- model picks **hold**

Adding auto-reject (the worst option) flips the preference from
phone-screen to hold. This is an IIA violation --- the compromise
effect in action. The model gravitates toward the "middle" of
whatever menu it sees.
