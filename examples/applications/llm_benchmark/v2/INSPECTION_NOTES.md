# v2 Manual Inspection Notes

## IIA Violation Patterns

### Pattern 1: "Weak option as anchor" (most common)
Adding a clearly-inferior third option changes the ranking between two competitive options.

**Job screen v04 × minimal**: phone_screen > hold_for_review in pairwise. But in {auto_reject, hold, phone_screen}, hold wins. Adding auto_reject (clearly worst) makes hold look better relative to phone_screen.

**Procurement v04 × minimal**: request_quotes > approve_with_tag in pairwise. But in {auto_approve, approve_with_tag, request_quotes}, approve_with_tag wins. Adding auto_approve (too permissive) makes the moderate option more attractive.

**Both procurement cases (v04 minimal, v06 aggressive)**: Identical IIA pattern on the same pair. The approve_with_tag vs request_quotes preference flips when auto_approve is added. Robust across prompts.

### Pattern 2: "Severity anchor" in content moderation
**Content v01 × aggressive**: content_warning > hide_from_feed in pairwise. But in {approve, content_warning, hide}, hide wins. Adding "approve" (too lenient) makes the LLM shift toward stricter action.

**Content v01 × decision_tree**: remove_and_strike > suspend_and_legal in pairwise. But in {approve, remove, suspend}, suspend wins. Adding "approve" pushes toward the most severe option.

**Content v03 × CoT**: approve > remove_and_strike in pairwise. But in {approve, remove, suspend}, remove wins. Adding "suspend" (extreme) makes "remove" look moderate and therefore preferable to "approve."

Pattern: in content moderation, adding a lenient option pushes toward strictness, and adding an extreme option pushes toward moderation. The LLM "anchors" to the extremes of the menu.

### Pattern 3: "Escalation anchor" in job screening
**Job v05 × decision_tree**: auto_reject > technical_interview in pairwise. But in {auto_reject, technical, fast_track}, technical wins. Adding fast_track (highest) makes technical look reasonable.

**Job v08 × conservative**: hold > phone_screen in pairwise. But in {auto_reject, hold, phone_screen}, phone_screen wins. Adding auto_reject (lowest) makes phone_screen look reasonable.

Both directions: adding a higher option pushes choice down, adding a lower option pushes choice up. The LLM gravitates toward the "middle" of whatever menu is shown — a compromise effect.

## SARP Pass Cases: What Clean Consistency Looks Like

**Support v00 × minimal** (clear): "Data deleted, business disruption." Clean linear ranking: escalate > account_mgr > bug_ticket > auto_reply > billing. Never wavers regardless of menu composition.

**Alert v03 × minimal** (binary): "CPU spike, slow response." Clean ranking: page_oncall > open_incident > execute_runbook > create_p3 > auto_resolve. Even with a "binary" vignette targeting the oncall/incident pair, the ranking holds.

## Parse Failures: A Signal, Not Noise

**Content v01 × decision_tree**: 5/15 menus return PARSE_FAIL — all menus containing ONLY mild options {approve, content_warning, hide_from_feed}. The LLM refuses to pick any mild action for graphic animal cruelty. It outputs something like "remove_and_strike" even though that's not in the menu. This is a *policy constraint* overriding the menu constraint — the LLM has a floor on severity for extreme content.

## Key Observations

1. **IIA violations cluster on "adjacent-severity" action pairs.** The pairs that flip are always two actions that are close in severity (hold/phone_screen, approve_with_tag/request_quotes, content_warning/hide_from_feed). Widely separated pairs (auto_reject vs fast_track) never flip.

2. **The compromise effect is the dominant mechanism.** Adding an extreme option to a menu pushes the choice toward the moderate option. This is well-documented in human choice (Simonson 1989) and now confirmed in LLMs.

3. **Parse failures on content review are informative.** The LLM's refusal to pick a mild action for severe content is itself a preference signal — it violates SARP but reveals a hard constraint in the model's policy.

4. **Alert triage SARP passes even on adversarial vignettes** because the severity ordering (auto_resolve < p3_ticket < page_oncall < incident_channel) is deeply embedded. Only execute_runbook creates ambiguity (it's not on the same severity axis).
