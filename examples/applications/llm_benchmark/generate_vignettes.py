#!/usr/bin/env python3
"""Stage 1: Generate realistic vignettes for each benchmark scenario.

Uses o4-mini to create diverse, realistic input vignettes for each scenario.
Menus (subsets of items) are generated deterministically and stored alongside.

Usage:
    python -m llm_benchmark.generate_vignettes --all
    python -m llm_benchmark.generate_vignettes --scenario support_ticket
    python -m llm_benchmark.generate_vignettes --scenario alert_triage --trials 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from .config import ALL_SCENARIOS, ScenarioConfig

DATA_DIR = Path(__file__).parent / "data" / "vignettes"


def generate_menus(n_items: int, n_trials: int, seed: int = 42) -> list[list[int]]:
    """Generate menus with guaranteed pairwise coverage.

    First C(n,2) menus cover all pairs. Rest are random subsets of size 2-4.
    """
    rng = np.random.default_rng(seed)
    menus: list[list[int]] = [sorted(pair) for pair in combinations(range(n_items), 2)]
    rng.shuffle(menus)  # type: ignore[arg-type]

    for _ in range(n_trials - len(menus)):
        size = int(rng.integers(2, min(5, n_items + 1)))
        items = rng.choice(n_items, size=size, replace=False)
        menus.append(sorted(items.tolist()))

    return menus[:n_trials]


def load_existing_vignettes(path: Path) -> dict[int, dict]:
    """Load already-generated vignettes, keyed by trial index."""
    existing: dict[int, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                existing[record["trial"]] = record
    return existing


def generate_vignettes_for_scenario(
    scenario: ScenarioConfig,
    n_trials: int = 200,
    seed: int = 42,
    batch_size: int = 20,
) -> None:
    """Generate vignettes for a single scenario using o4-mini."""
    from openai import OpenAI

    client = OpenAI()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{scenario.name}.jsonl"

    # Load existing and compute what's missing
    existing = load_existing_vignettes(out_path)
    menus = generate_menus(len(scenario.items), n_trials, seed)
    missing = [i for i in range(n_trials) if i not in existing]

    if not missing:
        print(f"  {scenario.name}: all {n_trials} vignettes already exist, skipping")
        return

    print(f"  {scenario.name}: generating {len(missing)} vignettes ({len(existing)} cached)")

    # Generate in batches
    for batch_start in range(0, len(missing), batch_size):
        batch_indices = missing[batch_start : batch_start + batch_size]
        n_batch = len(batch_indices)

        prompt = (
            f"{scenario.vignette_generation_prompt}\n\n"
            f"Generate exactly {n_batch} vignettes as a JSON array. "
            f"Each element should be a JSON object with a single key \"vignette\" "
            f"containing the 2-4 sentence scenario text.\n\n"
            f"Return ONLY the JSON array, no other text."
        )

        try:
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8000,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
                text = text.strip()

            vignettes = json.loads(text)
        except Exception as e:
            print(f"    ERROR generating batch at {batch_start}: {e}")
            continue

        # Write each vignette with its pre-assigned menu
        with open(out_path, "a") as f:
            for idx_in_batch, trial_idx in enumerate(batch_indices):
                if idx_in_batch >= len(vignettes):
                    break
                record = {
                    "trial": trial_idx,
                    "vignette": vignettes[idx_in_batch]["vignette"],
                    "menu": menus[trial_idx],
                }
                f.write(json.dumps(record) + "\n")
                existing[trial_idx] = record

        done = len(existing)
        print(f"    batch {batch_start // batch_size + 1}: "
              f"{done}/{n_trials} vignettes complete")

        time.sleep(0.5)  # rate limit

    print(f"  {scenario.name}: done - {len(existing)}/{n_trials} vignettes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark vignettes")
    parser.add_argument("--scenario", type=str, help="Single scenario name")
    parser.add_argument("--all", action="store_true", help="Generate for all scenarios")
    parser.add_argument("--trials", type=int, default=200, help="Vignettes per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for menus")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    if args.all:
        scenarios = list(ALL_SCENARIOS.values())
    elif args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(ALL_SCENARIOS)}")
            return
        scenarios = [ALL_SCENARIOS[args.scenario]]
    else:
        parser.print_help()
        return

    print(f"Generating vignettes for {len(scenarios)} scenario(s), {args.trials} trials each\n")
    for scenario in scenarios:
        generate_vignettes_for_scenario(scenario, args.trials, args.seed)
        print()


if __name__ == "__main__":
    main()
