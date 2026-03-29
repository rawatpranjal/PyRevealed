Case Study 1: Detecting Inconsistency in AI Agents
==================================================

Do LLMs have stable action rankings, or does the ranking change when
different alternatives are shown? We build preference graphs from LLM
decisions and check for cycles.

**TL;DR.** GPT-4o-mini operates with stable logical rankings for most tasks but
struggles with context-dependent framing. Between 74% to 92% of scenarios display
perfect logical consistency (SARP) at temperature 0. Decisions are
highly stable in domains like **Alert Triage** (92% pass rate), where showing
or hiding intermediate routing options rarely causes it to contradict its core
logic. In contrast, the **Jobs Task** is the weakest category (74% pass rate)
with frequent violations of menu independence (IIA)—for
example, the LLM might prefer "Interview" over "Reject" natively, but
introducing "Waitlist" as a third option inexplicably flips its choice to
"Reject". We test this logical consistency natively, evaluating probabilistic
choice (temperature > 0) strictly through Random Utility Models (RUM) rather
than mathematically flawed "majority votes".

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
   * - **Support**
     - auto-reply KB, bug ticket, billing, account mgr, escalate VP
     - **minimal** (bare instruction), **decision-tree** (if/then rules),
       **conservative** (prefer escalation), **aggressive** (prefer automation),
       **chain-of-thought** (numbered reasoning)
   * - **Alert**
     - auto-resolve, P3 ticket, page on-call, incident channel, runbook
     - same 5 prompts, adapted per scenario
   * - **Content Moderation Task**
     - approve, warning, hide, remove+strike, suspend+legal
     - same 5
   * - **Jobs Task**
     - reject, hold, phone screen, technical, fast-track
     - same 5
   * - **Procurement**
     - auto-approve, tag, request quotes, escalate, deny
     - same 5

.. _llm-how-to-read:

How to read the results
-----------------------

**Perfect Consistency (SARP)** checks if the model operates using a single, strict ranking of actions. If it prefers Action A over B, and B over C, it must logically prefer A over C. If it navigates all menus without contradicting itself or forming cyclical loops, it passes.

**Menu Independence (IIA)** checks if adding a third option changes how the model feels about the first two. If it picks "Interview" over "Reject" in a pair, but switches to "Reject" when "Waitlist" is added to the menu, the result is dependent on the menu framing and independence is violated.

**Probabilistic Consistency (RUM)** evaluates 20 samples per menu (at temperature 0.7) as a probability distribution. Instead of demanding a single fixed answer, it tests whether the model\'s varied responses can be explained by a logical distribution of preferences rather than random noise.

.. _llm-why-design:

Why this design
---------------

Holding the vignette constant and only changing the menu isolates menu
effects.

.. _llm-det-results:

Results 1: Deterministic (temp=0)
---------------------------------

First, we measure simple deterministic consistency. Each cell shows the percentage of configurations where the LLM behaves perfectly logically with no contradictions (passing SARP).

.. list-table:: SARP Pass Rate by Prompt Framework (%)
   :header-rows: 1
   :widths: 18 13 13 13 13 13 13

   * - Scenario
     - Minimal
     - Decision Tree
     - Conservative
     - Aggressive
     - Chain-of-Thought
     - Mean
   * - **Support**
     - 90
     - 80
     - 100
     - 80
     - 90
     - 88
   * - **Alert**
     - 80
     - 100
     - 90
     - 100
     - 90
     - 92
   * - **Content Moderation Task**
     - 90
     - 80
     - 70
     - 90
     - 80
     - 82
   * - **Jobs Task**
     - 70
     - 60
     - 80
     - 80
     - 80
     - 74
   * - **Procurement**
     - 70
     - 100
     - 70
     - 90
     - 90
     - 84

Prompt engineering significantly impacts consistency. Structuring decisions with 
rigid rules (Decision Tree) or directional bias (Conservative/Aggressive) 
tends to out-perform unguided Minimal prompts across most operational domains.

.. list-table:: SARP Pass Rate by Case Difficulty (%)
   :header-rows: 1
   :widths: 18 16 16 16 16 16

   * - Scenario
     - Clear Cases
     - Binary Cases
     - Ambiguous
     - Adversarial
     - Mean
   * - **Support**
     - 87
     - 93
     - 90
     - 80
     - 88
   * - **Alert**
     - 93
     - 100
     - 90
     - 80
     - 92
   * - **Content Moderation Task**
     - 60
     - 80
     - 100
     - 100
     - 82
   * - **Jobs Task**
     - 87
     - 67
     - 70
     - 70
     - 74
   * - **Procurement**
     - 93
     - 73
     - 90
     - 80
     - 84

Model consistency does not uniformly degrade on harder tasks. Paradoxically, 
**Content Moderation Task** decisions are less consistent on ostensibly "Clear" cases
(60%), while the **Jobs Task** struggles most on simplified "Binary" rulesets (67%).

.. _llm-iia:

.. list-table:: Independence of Irrelevant Alternatives (IIA) Flips
   :header-rows: 1
   :widths: 50 50

   * - Operational Scenario
     - Reversal Frequency (IIA Count)
   * - **Support**
     - 3
   * - **Alert**
     - 2
   * - **Content Moderation Task**
     - 9
   * - **Jobs Task**
     - 15
   * - **Procurement**
     - 8

*IIA violation = adding a third option inexplicably reverses the preference between two existing options.*

The **Jobs Task** and **Content Moderation Task** categories exhibit the highest susceptibility to context-dependent preference reversals, indicating that the mere presence of decoy options can systematically manipulate the LLM's logical alignment.

.. _llm-stoch-results:

.. _llm-rum:

Results 2: Stochastic Choice (RUM)
----------------------------------

Instead of testing single decisions, we ask the model 20 times per menu and convert the responses into choice frequencies. We then test if this distribution of choices is logically consistent. Do its probabilities stem from a stable distribution of preferences, or are they just random noise?

.. list-table:: RUM Pass Rate by Prompt Framework (%)
   :header-rows: 1
   :widths: 18 13 13 13 13 13

   * - Scenario
     - Minimal
     - Decision Tree
     - Conservative
     - Aggressive
     - Chain-of-Thought
   * - **Support**
     - 50
     - 30
     - 70
     - 50
     - 70
   * - **Alert**
     - 70
     - 60
     - 90
     - 70
     - 80
   * - **Content Moderation Task**
     - 50
     - 60
     - 60
     - 80
     - 50
   * - **Jobs Task**
     - 70
     - 40
     - 60
     - 80
     - 60
   * - **Procurement**\*
     - 67
     - 88
     - 50
     - 38
     - 63

Stochastic robustness requires tailored instructional frameworks. Decision Tree paths 
prove remarkably effective for Procurement, whereas **Jobs Task** pipelines align better with 
Aggressive pacing.

.. list-table:: RUM Pass Rate by Case Difficulty (%)
   :header-rows: 1
   :widths: 18 16 16 16 16

   * - Scenario
     - Clear Cases
     - Binary Cases
     - Ambiguous
     - Adversarial
   * - **Support**
     - 53
     - 67
     - 40
     - 50
   * - **Alert**
     - 67
     - 93
     - 70
     - 60
   * - **Content Moderation Task**
     - 13
     - 80
     - 70
     - 90
   * - **Jobs Task**
     - 80
     - 60
     - 50
     - 50
   * - **Procurement**\*
     - 47
     - 47
     - 100
     - 100

Stochastic consistency testing reinforces the deterministic findings. Procurement paradoxically performs perfectly under Ambiguous and Adversarial constraints, while seemingly clear-cut **Content Moderation Task** fractures into inconsistent distributions. The **Jobs Task**, conversely, correctly performs best on Clear cases but degrades under complex scenarios. 
(*Results with asterisks are based on partial procurement stage2 coverage.*)

.. _llm-patterns:

Patterns
--------

IIA violations are not evenly distributed. Decision-tree is the IIA
hotspot on the **Jobs Task** (accounting for nearly half of the category's violations), while it
scores 100% SARP on **Procurement**. Conservative leads on the **Content Moderation Task**
moderation (accounting for nearly half of the category's violations). The same prompt can be the most and least
consistent depending on the decision domain.

.. list-table:: IIA Violations by Framework (Top Entries)
   :header-rows: 1
   :widths: 22 22 16

   * - Scenario
     - Prompt Framework
     - Flips (IIA Count)
   * - **Jobs Task**
     - Decision Tree
     - 6
   * - **Content Moderation Task**
     - Conservative
     - 4
   * - **Content Moderation Task**
     - Decision Tree
     - 3
   * - **Content Moderation Task**
     - Aggressive
     - 2
   * - **Jobs Task**
     - Minimal
     - 2
   * - **Procurement**
     - Minimal
     - 2
   * - **Procurement**
     - Conservative
     - 2

**Compromise effect (Jobs Task).** In most **Jobs Task** IIA cases
the mechanism is the same. Adding an extreme option shifts the choice
toward the middle option. For example, the model prefers
*phone_screen* over *hold_for_review* in a pair. When shown the triple
{auto_reject, hold, phone_screen}, the choice moves to *hold*. The
reverse also appears. Adding *fast_track* can move a choice from
*auto_reject* to *technical_interview*. This is the familiar
compromise pattern.

**Severity anchor (Content Moderation Task).** Adding a lenient option pushes
the choice toward a stricter action, and adding an extreme option pushes
the choice toward a moderate action. For example, the model prefers
*remove_and_strike* over *suspend_and_legal* in a pair. When shown the
triple {approve, remove, suspend}, *suspend* wins. In the opposite
direction, adding *suspend* can make *remove* preferable to *approve*.
The model anchors to the extremes of the menu.

**Parse failures as a policy floor.** In severe **Content Moderation Task** cases, all-mild
menus sometimes return PARSE_FAIL. The model refuses to pick a mild
action and outputs a severe action instead, even though it is not in the
menu. This is not random error. It reveals a safety floor that the model
will not cross. SARP counts this as a violation, but it is better read
as a constraint revealed by the model.

.. _llm-findings:

Findings
--------

The **Jobs Task** is the least consistent scenario. The deterministic SARP
pass rate is 74 percent and the stochastic rate is 78 percent. We count
15 deterministic and 14 stochastic IIA violations. In practice this means
that adding a third candidate often changes which of two candidates is
preferred, a hallmark of the compromise pattern.

The **Content Moderation Task** shows that clear cases are not always stable. The
clear tier passes only 47 percent of the time under stochastic sampling
while ambiguous and adversarial tiers are much higher. Menu dependent
severity judgments remain even under probabilistic sampling. We still see persistent probability shifts indicating IIA violations, which confirms that context effects remain strong even when we aggregate.

Prompt effects are real and specific to the domain. Decision tree is the
only prompt to reach 100 percent on a scenario and it does so on
procurement. The same prompt scores 60 percent on the **Jobs Task**. Conservative
steadies support at 90 percent but scores 60 percent on the **Content Moderation Task**.
There is no universal best prompt. Alert triage is the most consistent
scenario at 90 percent under stochastic SARP, likely because the action
set has a clear ordinal severity ladder. Procurement with the
conservative prompt has 24 percent mixed menus, the highest of any
scenario and prompt. Spending authority choices are more sensitive to
sampling temperature when the prompt emphasizes caution.

Finally, stochastic sampling exhibits strongly peaked distributions. Only
8 to 12 percent of menus produce mixed responses across 20 repetitions, meaning
the language model stays highly confident in its ranking, even when it is
contradicting itself under different menu contexts. The inconsistency we see
is structural rather than noise.

.. _llm-cost:

Computational Cost
------------------

Measured on Apple M-series (11 cores). The analysis runs SARP + HM on each (vignette, prompt) configuration. 5 scenarios × 10 vignettes × 5 prompts = 250 configurations, of which 150 have sufficient data (≥ 5 menu observations each).

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15

   * - Stage
     - Configs
     - Wall Time
     - Peak Memory
   * - SARP + HM analysis
     - 150
     - 0.5 s
     - 19 MB

At 3.3 ms per configuration, the consistency analysis is negligible compared to the LLM API calls that generate the data (~3,750 calls for deterministic, ~75,000 for stochastic).

.. _llm-replication:

Replication
-----------

All vignettes and LLM responses are bundled in the repository — no API key needed to reproduce the analysis:

- `Vignettes (JSONL) <https://github.com/rawatpranjal/PrefGraph/tree/main/examples/applications/llm_benchmark/v2/data/vignettes>`_
- `Responses (JSONL) <https://github.com/rawatpranjal/PrefGraph/tree/main/examples/applications/llm_benchmark/v2/data/responses>`_
- `Results (JSON) <https://github.com/rawatpranjal/PrefGraph/tree/main/examples/applications/llm_benchmark/v2/data/results>`_

To regenerate from scratch (requires OpenAI API key):

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
   :widths: 24 12 64

   * - Metric
     - Range
     - Meaning
   * - Perfect Consistency (SARP)
     - 0--100%
     - % of configurations with completely logical, non-contradictory preferences
   * - Menu Independence (IIA)
     - 0--n
     - Frequency of context-dependent preference reversals
   * - Mixed Choices (%)
     - 0--100%
     - % of menus where probabilistic sampling yielded varied responses

Limitations
~~~~~~~~~~~

No ground truth (consistency ≠ accuracy). Synthetic vignettes. Single
model family. For theoretical details on stochastic rationality testing 
(regularity, Block-Marschak) see :doc:`../menu/theory_stochastic`.

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
