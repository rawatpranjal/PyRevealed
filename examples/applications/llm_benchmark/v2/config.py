"""v2 configuration: per-vignette SARP with fixed inputs and varied menus.

Key changes from v1:
  - Same 5 items per scenario (proven realistic in v1)
  - 10 curated vignettes per scenario (4 tiers: clear, binary, ambiguous, adversarial)
  - All C(5,2)=10 pairwise + 5 size-3 menus per vignette = 15 menus
  - gpt-4o-mini only (v1 showed models are indistinguishable on consistency)
  - Two-stage: deterministic screen → stochastic deep dive on failures
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

# Reuse v1 scenario configs - same items, same prompts
from ..config import (
    ALL_SCENARIOS,
    SUPPORT_TICKET,
    ALERT_TRIAGE,
    CONTENT_REVIEW,
    JOB_SCREEN,
    PROCUREMENT,
    ScenarioConfig,
)

MODEL = {"name": "gpt-4o-mini", "slug": "gpt4omini", "temperature": 0.0}
STOCHASTIC_TEMPERATURE = 0.7
STOCHASTIC_K = 20


def generate_menus_for_vignette(n_items: int = 5) -> list[list[int]]:
    """Generate all menus for one vignette: C(n,2) pairwise + size-3 triples.

    With 5 items: 10 pairwise + 5 triples = 15 menus.
    The triples are chosen so each item appears as the "third" option
    added to different pairwise menus → enables IIA testing.
    """
    items = list(range(n_items))

    # All pairwise menus
    pairwise = [sorted(pair) for pair in combinations(items, 2)]

    # Size-3 menus: one per item, adding that item to its first pairwise pair
    triples = []
    used_pairs = set()
    for item in items:
        # Find a pairwise menu that doesn't include this item
        for pair in pairwise:
            if item not in pair and tuple(pair) not in used_pairs:
                triples.append(sorted(pair + [item]))
                used_pairs.add(tuple(pair))
                break

    return pairwise + triples


@dataclass(frozen=True)
class VignetteSpec:
    """Specification for one curated vignette."""
    tier: str          # "clear", "binary", "ambiguous", "adversarial"
    description: str   # Human description of what this vignette tests
    target_pair: tuple[int, int] | None  # The competing pair (for binary/adversarial)


# Vignette generation prompts per tier
VIGNETTE_TIER_PROMPTS = {
    "clear": (
        "Generate a situation where ONE specific action is obviously the correct choice. "
        "The situation should be unambiguous - any reasonable person would pick the same action. "
        "Make it concrete and realistic (2-3 sentences)."
    ),
    "binary": (
        "Generate a situation that is genuinely ambiguous between exactly TWO of the actions. "
        "The situation should have features that make both actions plausible - reasonable "
        "people could disagree on which is better. Make it concrete (2-3 sentences). "
        "Target the competing pair: {pair_names}."
    ),
    "ambiguous": (
        "Generate a situation where THREE or more actions are plausible. The situation "
        "should be complex enough that multiple approaches are defensible. "
        "Make it concrete and realistic (2-3 sentences)."
    ),
    "adversarial": (
        "Generate a situation designed to trigger menu-composition effects. The situation "
        "should have features that make the preferred action CHANGE depending on what "
        "other options are available. For example: if only {pair_names} are offered, "
        "the first seems better; but if a third 'decoy' option is added, the second "
        "becomes more attractive. Make it subtle and realistic (2-3 sentences)."
    ),
}

# Competing pairs identified from v1 EDA (top pair per scenario)
V1_COMPETING_PAIRS = {
    "support_ticket": [(1, 3), (0, 3), (1, 4)],       # bug vs account_mgr, kb vs account_mgr, bug vs escalate
    "alert_triage": [(3, 2), (1, 2), (1, 3)],          # incident vs oncall, p3 vs oncall, p3 vs incident
    "content_review": [(0, 2), (0, 3), (1, 2)],        # approve vs hide, approve vs remove, warning vs hide
    "job_screen": [(1, 2), (2, 3), (4, 2)],            # hold vs phone, phone vs technical, fast vs phone
    "procurement": [(1, 3), (0, 3), (1, 2)],           # tag vs escalate, approve vs escalate, tag vs quotes
}

# Tier distribution: 3 clear + 3 binary + 2 ambiguous + 2 adversarial = 10
VIGNETTE_TIERS = [
    ("clear", None),
    ("clear", None),
    ("clear", None),
    ("binary", 0),      # target competing pair index 0
    ("binary", 1),      # target competing pair index 1
    ("binary", 2),      # target competing pair index 2
    ("ambiguous", None),
    ("ambiguous", None),
    ("adversarial", 0),  # target competing pair index 0
    ("adversarial", 1),  # target competing pair index 1
]
