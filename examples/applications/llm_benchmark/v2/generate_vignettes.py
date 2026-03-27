#!/usr/bin/env python3
"""v2 Stage 1: Generate curated vignettes for per-vignette SARP testing.

Generates 10 vignettes per scenario, curated across 4 difficulty tiers.
Each vignette gets ALL 15 menus (10 pairwise + 5 triple).

Usage:
    python -m applications.llm_benchmark.v2.generate_vignettes --all
    python -m applications.llm_benchmark.v2.generate_vignettes --scenario support_ticket
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ..config import ALL_SCENARIOS
from .config import (
    VIGNETTE_TIERS,
    VIGNETTE_TIER_PROMPTS,
    V1_COMPETING_PAIRS,
    generate_menus_for_vignette,
)

DATA_DIR = Path(__file__).parent / "data" / "vignettes"


def generate_vignettes_for_scenario(scenario_name: str) -> None:
    """Generate 10 curated vignettes for one scenario."""
    from openai import OpenAI

    client = OpenAI()
    scenario = ALL_SCENARIOS[scenario_name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{scenario_name}.jsonl"

    # Check existing
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                existing[r["vignette_id"]] = r
    if len(existing) >= 10:
        print(f"  {scenario_name}: all 10 vignettes exist, skipping")
        return

    menus = generate_menus_for_vignette(len(scenario.items))
    competing_pairs = V1_COMPETING_PAIRS.get(scenario_name, [(0, 1), (1, 2), (2, 3)])

    items_desc = "\n".join(
        f"  {idx}: {name} — {scenario.item_descriptions[idx]}"
        for idx, name in sorted(scenario.items.items())
    )

    with open(out_path, "a") as f:
        for vig_idx, (tier, pair_idx) in enumerate(VIGNETTE_TIERS):
            vig_id = f"{scenario_name}_v{vig_idx:02d}_{tier}"
            if vig_id in existing:
                continue

            # Build the generation prompt
            tier_prompt = VIGNETTE_TIER_PROMPTS[tier]
            if pair_idx is not None and "{pair_names}" in tier_prompt:
                pair = competing_pairs[pair_idx % len(competing_pairs)]
                pair_names = f"{scenario.items[pair[0]]} and {scenario.items[pair[1]]}"
                tier_prompt = tier_prompt.format(pair_names=pair_names)

            prompt = (
                f"Context: You are generating test inputs for a {scenario.display_name} system.\n"
                f"The system must choose one of these actions:\n{items_desc}\n\n"
                f"{tier_prompt}\n\n"
                f"Return ONLY the situation text (2-3 sentences), nothing else."
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=200,
                )
                vignette_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"    ERROR on {vig_id}: {e}")
                continue

            record = {
                "vignette_id": vig_id,
                "vignette_idx": vig_idx,
                "tier": tier,
                "target_pair": list(competing_pairs[pair_idx % len(competing_pairs)]) if pair_idx is not None else None,
                "vignette": vignette_text,
                "menus": menus,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            existing[vig_id] = record
            print(f"    {vig_id}: \"{vignette_text[:80]}...\"")
            time.sleep(0.3)

    print(f"  {scenario_name}: {len(existing)}/10 vignettes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 curated vignettes")
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    scenarios = list(ALL_SCENARIOS) if args.all else ([args.scenario] if args.scenario else [])
    if not scenarios:
        parser.print_help()
        return

    print(f"Generating v2 vignettes for {len(scenarios)} scenario(s)\n")
    for sc in scenarios:
        generate_vignettes_for_scenario(sc)
        print()


if __name__ == "__main__":
    main()
