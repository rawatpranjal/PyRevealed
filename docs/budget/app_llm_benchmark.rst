LLM Consistency
===============

Do LLMs have stable action rankings, or does the ranking change when
different alternatives are shown? We build preference graphs from LLM
decisions and check for cycles.

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

.. code-block:: text

   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │ Vignettes  │───▶│ Query LLM  │───▶│ Build pref │───▶│ Test SARP  │
   │ (10 per    │    │ (15 menus  │    │ graph per  │    │ + IIA per  │
   │ scenario)  │    │ per vig.)  │    │ vignette   │    │ vignette   │
   └────────────┘    └────────────┘    └────────────┘    └────────────┘

Setup
-----

A **scenario** is a real enterprise decision task (support triage,
alert routing, etc.). Each scenario has 5 **actions** the LLM can
choose from. A **vignette** is one concrete input --- a specific support
ticket, alert payload, or candidate resume --- that the LLM must
respond to. A **prompt** is the system-prompt persona framing how the
LLM should approach the decision. A **menu** is a subset of the 5
actions shown to the LLM for a given vignette; we present all
C(5,2)=10 pairwise menus + 5 size-3 menus = 15 menus per vignette.

5 scenarios, 10 vignettes each, 5 prompts, 15 menus per vignette.
Model: gpt-4o-mini. Deterministic (temp=0) and stochastic (temp=0.7,
K=20 reps per menu).

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Scenario
     - Actions
   * - Support
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
   * - Alert
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
   * - Content
     - approve, warning, hide, remove+strike, suspend+legal
   * - Jobs
     - reject, hold, phone screen, technical, fast-track
   * - Procurement
     - auto-approve, tag, request quotes, escalate, deny

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Prompt
     - Framing
   * - minimal
     - One-line instruction with no guidance ("Route support tickets.")
   * - decision-tree
     - Explicit if/then rules mapping input features to actions
   * - conservative
     - Risk-averse persona that prefers escalation over automation
   * - aggressive
     - Efficiency-first persona that prefers automation over escalation
   * - chain-of-thought
     - Numbered reasoning steps (identify intent, assess urgency, etc.)

How to read the results
-----------------------

- SARP is a deterministic consistency check. We build a preference graph
  by adding an edge from the chosen item to every unchosen item in the
  same menu. SARP passes when the transitive closure of this graph has
  no cycles, which is equivalent to having a strict ranking that explains
  all choices.
- IIA violations are detected by comparing pairwise menus with the
  corresponding triples. If A beats B in the pair {A, B}, but adding C
  shifts the choice to B in {A, B, C}, then the result depends on the
  menu and independence is violated.
- Stochastic results use K=20 samples at temperature 0.7 per menu and
  report the majority choice. Agreement measures the percent of menus
  where this majority matches the deterministic pick. Percent mixed is
  the share of menus where the K responses do not all agree.

Why this design
---------------

Holding the vignette constant and only changing the menu isolates menu
effects. Earlier designs that varied both content and menus at the same
time made it hard to tell whether a flip was caused by a different story
or a different set of options. This setup pins flips on the menu.

Deterministic Results (temp=0)
------------------------------

Each cell = % of 10 vignettes where the preference graph is acyclic.

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

Stochastic Results (temp=0.7, K=20)
------------------------------------

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

Findings
--------

- **Job screening is the least consistent scenario.** 74 percent
  deterministic SARP pass and 78 percent stochastic. There are 15
  deterministic and 14 stochastic IIA violations. Adding a third
  candidate changes which of two is preferred.
- **Content moderation "clear" vignettes pass only 47 percent
  stochastically.** Even unambiguous posts produce menu dependent
  severity judgments. There are 12 stochastic IIA violations. Majority
  voting does not eliminate these context effects.
- **Decision tree is the only prompt to hit 100 percent on any
  scenario** (procurement stochastic). It scores 60 percent on jobs.
  Conservative scores 90 percent on support and 60 percent on content.
  There is no universal best prompt.
- **Alert triage is the most consistent** (90 percent stochastic SARP).
  Actions have a clear ordinal severity structure that resists menu
  effects.
- **Procurement conservative prompt has 24 percent mixed menus,** the
  highest of any scenario and prompt. Spending authority decisions are
  sensitive to sampling temperature when the prompt emphasizes caution.
- **Stochastic sampling barely changes rankings.** 96 to 98 percent of
  menus agree between temperature 0 and temperature 0.7 majority vote.
  Only 8 to 12 percent of menus produce mixed responses across 20
  repetitions. Inconsistency is structural, not noise.

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
