"""Scenario and model configurations for the LLM Enterprise Consistency Benchmark.

Each scenario represents a real-world LLM deployment endpoint. Each has:
  - 5 discrete actions the system can take
  - 5 production-grade system prompts (prompt engineering A/B test variants)
  - A vignette generation prompt for creating realistic test inputs
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for one benchmark scenario."""

    name: str                              # slug: "support_ticket"
    display_name: str                      # "SaaS Support Ticket Router"
    items: dict[int, str]                  # {0: "auto_reply_kb", ...}
    item_descriptions: dict[int, str]      # longer descriptions for user prompt
    context: str                           # task framing for the user prompt
    system_prompts: dict[str, str]         # {"minimal": "...", ...}
    vignette_generation_prompt: str        # prompt for o4-mini to generate vignettes


MODEL_CONFIGS = [
    {"name": "gpt-4o-mini", "slug": "gpt4omini", "temperature": 0.0},
    {"name": "o4-mini", "slug": "o4mini"},  # reasoning model, no temperature param
]


# =============================================================================
# Scenario 1: SaaS Support Ticket Router
# =============================================================================

SUPPORT_TICKET = ScenarioConfig(
    name="support_ticket",
    display_name="SaaS Support Ticket Router",
    items={
        0: "auto_reply_kb",
        1: "create_bug_ticket",
        2: "route_billing",
        3: "route_account_mgr",
        4: "escalate_vp",
    },
    item_descriptions={
        0: "Auto-reply with knowledge base article link",
        1: "Create engineering bug ticket in Jira",
        2: "Route to billing specialist queue",
        3: "Route to dedicated account manager",
        4: "Escalate to VP of Support (urgent)",
    },
    context=(
        "You are the automated triage system for a B2B SaaS company's support inbox. "
        "Each incoming ticket must be routed to exactly one destination. "
        "Read the customer's message and choose the single best action from the options below. "
        "Respond with ONLY the action name, nothing else."
    ),
    system_prompts={
        "minimal": (
            "Route support tickets. Pick one action from the list."
        ),
        "decision_tree": (
            "You are the Tier-1 triage system for Acme Cloud's support inbox. "
            "Route each ticket using these rules:\n"
            "- If the customer describes a bug, error message, or unexpected behavior → create_bug_ticket\n"
            "- If the message mentions invoices, charges, subscription, payment, or pricing → route_billing\n"
            "- If the customer is an enterprise account asking about their contract, onboarding, "
            "or requesting a feature → route_account_mgr\n"
            "- If the message mentions 'production down', 'outage', 'SLA breach', or 'legal' → escalate_vp\n"
            "- For general questions, how-to requests, or known FAQ topics → auto_reply_kb\n"
            "When in doubt between two options, prefer the more specific route over auto_reply_kb.\n"
            "Respond with ONLY the action name."
        ),
        "conservative": (
            "You are a careful support triage system. Your top priority is ensuring no customer "
            "falls through the cracks. When the intent is ambiguous, always prefer routing to a human "
            "(route_account_mgr or escalate_vp) over an automated response (auto_reply_kb). "
            "Err on the side of over-escalation - a false escalation costs 5 minutes of a human's time, "
            "but a missed urgent issue can cost a customer. "
            "If the customer sounds frustrated or mentions impact on their business, escalate. "
            "Respond with ONLY the action name."
        ),
        "aggressive": (
            "You are an efficiency-optimized support triage system. Your goal is to minimize human "
            "involvement and resolve tickets automatically wherever possible. "
            "Use auto_reply_kb for any question that could plausibly be answered by documentation. "
            "Use create_bug_ticket only for clear, reproducible bugs with specific error details. "
            "Only route to humans (route_billing, route_account_mgr) when automation genuinely cannot help. "
            "Reserve escalate_vp exclusively for situations involving active service outages affecting "
            "multiple customers or explicit legal threats. "
            "Respond with ONLY the action name."
        ),
        "chain_of_thought": (
            "You are a support triage system. For each ticket, follow this reasoning process:\n"
            "1. Identify the customer's primary intent in one sentence.\n"
            "2. Assess urgency: is this time-sensitive (hours) or can it wait (days)?\n"
            "3. Determine if this can be resolved by documentation alone.\n"
            "4. Check if this involves money, contracts, or account-level changes.\n"
            "5. Based on your analysis, choose the single best action.\n\n"
            "Think through steps 1-4 internally, then respond with ONLY the action name."
        ),
    },
    vignette_generation_prompt=(
        "Generate realistic B2B SaaS support tickets. Each should be 2-3 sentences from a customer "
        "of a cloud analytics platform. Include a mix of:\n"
        "- Bug reports (dashboard errors, data sync issues, API failures)\n"
        "- Billing questions (invoice discrepancies, plan upgrades, refund requests)\n"
        "- Feature requests and onboarding help from enterprise accounts\n"
        "- Urgent production issues (outages, data loss, SLA concerns)\n"
        "- General how-to questions answerable by documentation\n"
        "Make them realistic and ambiguous - some tickets should be borderline between categories. "
        "Vary the customer tone from calm to frustrated to panicked."
    ),
)


# =============================================================================
# Scenario 2: Cloud Infrastructure Alert Triage
# =============================================================================

ALERT_TRIAGE = ScenarioConfig(
    name="alert_triage",
    display_name="Cloud Infrastructure Alert Triage",
    items={
        0: "auto_resolve",
        1: "create_p3_ticket",
        2: "page_oncall",
        3: "open_incident_channel",
        4: "execute_runbook",
    },
    item_descriptions={
        0: "Auto-resolve - known transient issue, no action needed",
        1: "Create P3 ticket for next sprint backlog",
        2: "Page the on-call engineer immediately",
        3: "Open incident channel and page team lead",
        4: "Execute automated runbook (restart/scale/failover)",
    },
    context=(
        "You are the alert triage system for a cloud infrastructure team. "
        "A monitoring alert has fired. Based on the alert details, choose the single best "
        "response action from the options below. "
        "Respond with ONLY the action name, nothing else."
    ),
    system_prompts={
        "minimal": (
            "Triage infrastructure alerts. Pick one action from the list."
        ),
        "decision_tree": (
            "You are the alert triage system for Acme Cloud's infrastructure team. "
            "Route each alert using these rules:\n"
            "- CPU/memory spikes under 5 minutes with no error rate increase → auto_resolve\n"
            "- Disk usage warnings below 90%, gradual increase → create_p3_ticket\n"
            "- Error rate above 1% or latency p99 above 2x baseline → page_oncall\n"
            "- Multiple services affected OR customer-facing outage detected → open_incident_channel\n"
            "- Known failure pattern with documented fix (OOM restart, certificate rotation, "
            "connection pool exhaustion) → execute_runbook\n"
            "If the alert matches a known runbook AND is customer-facing, prefer open_incident_channel.\n"
            "Respond with ONLY the action name."
        ),
        "conservative": (
            "You are a cautious alert triage system. Missed pages during real incidents have "
            "historically cost the company $50K+ in SLA credits. Your bias should be toward "
            "paging humans rather than auto-resolving. "
            "Only auto_resolve alerts that are definitively known false positives (e.g., scheduled "
            "maintenance windows, known flaky tests). "
            "When in doubt between page_oncall and open_incident_channel, choose the more severe option. "
            "Waking someone up unnecessarily is better than missing a real incident. "
            "Respond with ONLY the action name."
        ),
        "aggressive": (
            "You are a noise-reduction-focused alert triage system. Alert fatigue is the #1 problem - "
            "the on-call team gets paged 40 times per week and most are false alarms. "
            "Auto-resolve any alert that has self-recovered within the last 3 occurrences. "
            "Use execute_runbook whenever a documented fix exists, even if you're not 100% sure it applies. "
            "Create P3 tickets for things that need investigation but aren't urgent. "
            "Only page humans for clear, ongoing customer-facing impact. "
            "Respond with ONLY the action name."
        ),
        "chain_of_thought": (
            "You are an alert triage system. For each alert, reason through:\n"
            "1. What service is affected and what is the failure mode?\n"
            "2. Is this self-healing (transient) or persistent?\n"
            "3. Is there customer-facing impact right now?\n"
            "4. Does a documented runbook exist for this failure pattern?\n"
            "5. Based on severity and automation options, choose the best action.\n\n"
            "Think through steps 1-4 internally, then respond with ONLY the action name."
        ),
    },
    vignette_generation_prompt=(
        "Generate realistic cloud infrastructure monitoring alerts. Each should be a structured "
        "alert payload in plain English, 2-3 sentences, including:\n"
        "- The service name and metric (e.g., 'api-gateway latency p99')\n"
        "- Current value vs threshold (e.g., '450ms, threshold 200ms')\n"
        "- Duration and trend (e.g., 'sustained for 12 minutes, increasing')\n\n"
        "Include a mix of: CPU/memory spikes, error rate increases, latency degradation, "
        "disk usage warnings, certificate expiration, connection pool exhaustion, "
        "deployment-related anomalies, database replication lag, and DNS resolution failures. "
        "Some should be obvious false alarms, some clearly critical, and many borderline."
    ),
)


# =============================================================================
# Scenario 3: Social Media Content Review
# =============================================================================

CONTENT_REVIEW = ScenarioConfig(
    name="content_review",
    display_name="Social Media Content Review",
    items={
        0: "approve",
        1: "content_warning",
        2: "hide_from_feed",
        3: "remove_and_strike",
        4: "suspend_and_legal",
    },
    item_descriptions={
        0: "Approve - false flag, content is fine",
        1: "Add content warning label (sensitive topic)",
        2: "Hide from recommendations feed (reduce reach)",
        3: "Remove post and issue account strike",
        4: "Suspend account and flag for legal review",
    },
    context=(
        "You are the content moderation system for a social media platform. "
        "A post has been flagged by users or automated filters. Review the post text and "
        "choose the single most appropriate moderation action from the options below. "
        "Respond with ONLY the action name, nothing else."
    ),
    system_prompts={
        "minimal": (
            "Moderate flagged social media posts. Pick one action from the list."
        ),
        "decision_tree": (
            "You are the content moderation system for a social platform. "
            "Apply these community guidelines:\n"
            "- Clearly protected speech (opinion, satire, news reporting) → approve\n"
            "- Discussions of sensitive topics (violence in news, medical procedures, "
            "political debate) without policy violation → content_warning\n"
            "- Borderline content that doesn't explicitly violate rules but could be harmful "
            "at scale (clickbait health claims, soft misinformation) → hide_from_feed\n"
            "- Clear policy violations (harassment, hate speech, graphic violence, "
            "dangerous challenges) → remove_and_strike\n"
            "- Credible threats, CSAM, coordinated inauthentic behavior, doxxing → suspend_and_legal\n"
            "Apply the least restrictive action that addresses the concern.\n"
            "Respond with ONLY the action name."
        ),
        "conservative": (
            "You are a safety-first content moderation system. Platform trust and user safety "
            "are the top priorities. The cost of leaving harmful content up is much higher than "
            "the cost of over-moderating. "
            "When a post is borderline, choose the more restrictive action. "
            "If content could be interpreted as harassment by a reasonable person, remove it. "
            "If you're unsure whether something is a credible threat, treat it as one. "
            "Only approve content that is clearly and unambiguously within guidelines. "
            "Respond with ONLY the action name."
        ),
        "aggressive": (
            "You are a free-expression-prioritizing content moderation system. Over-moderation "
            "destroys user trust and engagement. False positives (wrongly removed content) are "
            "worse than false negatives for platform health. "
            "Approve any content that has a reasonable interpretation as legitimate speech. "
            "Use content_warning and hide_from_feed as intermediate steps before removal. "
            "Reserve remove_and_strike only for unambiguous, severe policy violations. "
            "Reserve suspend_and_legal only for credible real-world threats or illegal content. "
            "Respond with ONLY the action name."
        ),
        "chain_of_thought": (
            "You are a content moderation system. For each flagged post, reason through:\n"
            "1. What type of content is this? (opinion, humor, news, personal attack, etc.)\n"
            "2. Does it explicitly violate any community guideline?\n"
            "3. Could it cause real-world harm if widely distributed?\n"
            "4. Is there a less restrictive action that adequately addresses the concern?\n"
            "5. Based on proportionality, choose the most appropriate action.\n\n"
            "Think through steps 1-4 internally, then respond with ONLY the action name."
        ),
    },
    vignette_generation_prompt=(
        "Generate realistic flagged social media posts for a content moderation queue. "
        "Each should be 2-4 sentences of actual post text (not a description of the post). "
        "Include a mix of:\n"
        "- Edgy humor and satire that could be misread\n"
        "- Health claims of varying accuracy\n"
        "- Political opinions and heated debate\n"
        "- Mild and severe profanity in various contexts\n"
        "- Personal conflicts that may or may not constitute harassment\n"
        "- News commentary on violent events\n"
        "- Product reviews with extreme language\n"
        "- Potential misinformation with partial truths\n"
        "Make them borderline and ambiguous - the kind of posts where reasonable people disagree."
    ),
)


# =============================================================================
# Scenario 4: Job Application Screen
# =============================================================================

JOB_SCREEN = ScenarioConfig(
    name="job_screen",
    display_name="Job Application Screen",
    items={
        0: "auto_reject",
        1: "hold_for_review",
        2: "phone_screen",
        3: "technical_interview",
        4: "fast_track",
    },
    item_descriptions={
        0: "Auto-reject - does not meet minimum requirements",
        1: "Hold for recruiter manual review",
        2: "Schedule initial phone screen",
        3: "Advance directly to technical interview",
        4: "Fast-track to final round (exceptional candidate)",
    },
    context=(
        "You are the application screening system for a mid-size tech company hiring "
        "a Senior Backend Engineer (Python, distributed systems, 4+ years experience). "
        "Review the candidate summary and choose the single best next step from the options below. "
        "Respond with ONLY the action name, nothing else."
    ),
    system_prompts={
        "minimal": (
            "Screen job applications for a Senior Backend Engineer role. "
            "Pick one action from the list."
        ),
        "decision_tree": (
            "You are the application screening system for a Senior Backend Engineer role. "
            "Requirements: Python proficiency, distributed systems experience, 4+ years. "
            "Screen using these criteria:\n"
            "- No relevant programming experience or < 2 years total → auto_reject\n"
            "- Relevant background but missing a key requirement (e.g., no Python, "
            "no distributed systems) → hold_for_review\n"
            "- Meets requirements, standard background → phone_screen\n"
            "- Strong match: relevant experience at notable companies, open source contributions, "
            "or advanced degree in CS → technical_interview\n"
            "- Exceptional: senior/staff at FAANG or top startup, major open source maintainer, "
            "published systems research → fast_track\n"
            "Respond with ONLY the action name."
        ),
        "conservative": (
            "You are a thorough application screening system. The cost of a bad hire ($150K+) "
            "far exceeds the cost of a slower pipeline. "
            "Be skeptical of inflated titles and vague descriptions of impact. "
            "If a resume lacks specific technical details, hold for review rather than advancing. "
            "Fast-track only candidates with independently verifiable exceptional achievements. "
            "When in doubt between two stages, choose the earlier one (e.g., phone_screen over "
            "technical_interview). "
            "Respond with ONLY the action name."
        ),
        "aggressive": (
            "You are a speed-optimized screening system. The talent market is competitive and "
            "top candidates accept offers within 5 days. "
            "Move promising candidates forward quickly - err on the side of advancing rather than "
            "holding. Use phone_screen as the default for anyone who meets basic requirements. "
            "Skip the phone screen (technical_interview) for candidates with clearly strong backgrounds. "
            "Fast-track anyone who would be a clear yes at the technical stage. "
            "Only auto_reject candidates who genuinely lack the minimum qualifications. "
            "Respond with ONLY the action name."
        ),
        "chain_of_thought": (
            "You are an application screening system. For each candidate, reason through:\n"
            "1. Does this candidate meet the minimum requirements (Python, distributed systems, 4+ yrs)?\n"
            "2. How strong is the signal from their experience (specific projects, scale, impact)?\n"
            "3. Are there any red flags (gaps, mismatched titles, vague descriptions)?\n"
            "4. How does this candidate compare to a typical qualified applicant for this role?\n"
            "5. Based on signal strength and risk, choose the appropriate pipeline stage.\n\n"
            "Think through steps 1-4 internally, then respond with ONLY the action name."
        ),
    },
    vignette_generation_prompt=(
        "Generate realistic candidate summaries for a Senior Backend Engineer role "
        "(Python, distributed systems, 4+ years). Each should be 3-5 sentences summarizing "
        "the candidate's background. Include a mix of:\n"
        "- Career switchers (data science → backend, frontend → backend)\n"
        "- Overqualified candidates (Staff Engineer at FAANG applying for senior)\n"
        "- Junior developers with impressive side projects\n"
        "- Exact-match candidates with standard backgrounds\n"
        "- Candidates with gaps or non-traditional paths\n"
        "- International candidates with experience at local companies\n"
        "- Bootcamp graduates with 2-3 years experience\n"
        "- Senior engineers from adjacent domains (DevOps, ML infra)\n"
        "Make them realistic - include specific technologies, company types, and years."
    ),
)


# =============================================================================
# Scenario 5: Procurement Approval
# =============================================================================

PROCUREMENT = ScenarioConfig(
    name="procurement",
    display_name="Procurement Approval",
    items={
        0: "auto_approve",
        1: "approve_with_tag",
        2: "request_quotes",
        3: "escalate_to_head",
        4: "deny",
    },
    item_descriptions={
        0: "Auto-approve - within policy, no review needed",
        1: "Approve with budget code tagging for tracking",
        2: "Request additional vendor quotes before approval",
        3: "Escalate to department head for review",
        4: "Deny request with explanation",
    },
    context=(
        "You are the procurement approval system for a 500-person tech company. "
        "A purchase request has been submitted. Review the details and choose the single "
        "best approval action from the options below. "
        "Respond with ONLY the action name, nothing else."
    ),
    system_prompts={
        "minimal": (
            "Process procurement requests. Pick one action from the list."
        ),
        "decision_tree": (
            "You are the procurement approval system. Process requests using these rules:\n"
            "- Recurring SaaS subscriptions under $500/month with existing contract → auto_approve\n"
            "- One-time purchases $500-$5,000 with clear business justification → approve_with_tag\n"
            "- Any purchase over $5,000 without competitive bids → request_quotes\n"
            "- Purchases over $25,000, new vendor relationships, or multi-year commitments "
            "→ escalate_to_head\n"
            "- Requests with no clear business justification, duplicate tools already in use, "
            "or from budget-frozen departments → deny\n"
            "When amount is ambiguous, use the more conservative action.\n"
            "Respond with ONLY the action name."
        ),
        "conservative": (
            "You are a fiscally conservative procurement system. Every dollar saved goes to "
            "runway, and the company is watching burn rate carefully. "
            "Default to requiring additional justification or quotes. "
            "Auto-approve only genuinely essential, recurring expenses. "
            "If a tool has a free tier or open-source alternative, recommend denial. "
            "Escalate anything with long-term cost implications (annual contracts, per-seat pricing "
            "that scales with headcount). "
            "Respond with ONLY the action name."
        ),
        "aggressive": (
            "You are a velocity-optimized procurement system. The biggest cost is engineer time "
            "wasted waiting for approvals. Move fast on tools that improve productivity. "
            "Auto-approve any SaaS tool under $2,000/month if the justification mentions "
            "productivity, efficiency, or developer experience. "
            "Approve with tagging rather than requesting quotes for amounts under $10,000. "
            "Only escalate or deny when there's a clear policy conflict or budget overrun. "
            "Respond with ONLY the action name."
        ),
        "chain_of_thought": (
            "You are a procurement approval system. For each request, reason through:\n"
            "1. What is being purchased and what is the total cost commitment?\n"
            "2. Is there a clear business justification tied to a team goal?\n"
            "3. Does this overlap with existing tools or contracts?\n"
            "4. Is this within the requester's typical spending authority?\n"
            "5. Based on cost, justification strength, and policy, choose the appropriate action.\n\n"
            "Think through steps 1-4 internally, then respond with ONLY the action name."
        ),
    },
    vignette_generation_prompt=(
        "Generate realistic procurement purchase requests for a 500-person tech company. "
        "Each should be 2-4 sentences including:\n"
        "- What is being purchased (SaaS tool, hardware, service, etc.)\n"
        "- Cost (monthly or one-time) and vendor name\n"
        "- Business justification from the requester\n"
        "- Department or team making the request\n\n"
        "Include a mix of: SaaS subscriptions ($20-$50K/year range), hardware purchases, "
        "conference travel, consulting engagements, training programs, infrastructure costs, "
        "duplicate tool requests, emergency purchases, and questionable nice-to-haves. "
        "Vary amounts from $50 to $100,000+. Make some clearly approvable, some clearly "
        "deniable, and many borderline."
    ),
)


# =============================================================================
# Registry
# =============================================================================

ALL_SCENARIOS: dict[str, ScenarioConfig] = {
    s.name: s for s in [SUPPORT_TICKET, ALERT_TRIAGE, CONTENT_REVIEW, JOB_SCREEN, PROCUREMENT]
}
