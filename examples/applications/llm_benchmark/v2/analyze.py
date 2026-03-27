#!/usr/bin/env python3
"""v2 Stage 3: Per-vignette SARP + IIA analysis.

For each (vignette, prompt): test SARP on 15 menu choices.
Detect IIA violations by comparing pairwise vs triple menus.

Usage:
    python -m applications.llm_benchmark.v2.analyze --all
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pyrevealed import MenuChoiceLog
from pyrevealed.algorithms.abstract_choice import validate_menu_sarp, compute_menu_efficiency

from ..config import ALL_SCENARIOS

RESPONSE_DIR = Path(__file__).parent / "data" / "responses"
RESULTS_DIR = Path(__file__).parent / "data" / "results"


def detect_iia_violations(records: list[dict], n_items: int = 5) -> list[dict]:
    """Detect IIA violations: does adding a third item flip a pairwise preference?

    Compare choice in {A,B} with the A-vs-B preference in {A,B,C}.
    """
    # Index records by menu (as tuple)
    by_menu = {}
    for r in records:
        by_menu[tuple(sorted(r["menu"]))] = r

    violations = []
    pairwise = {k: v for k, v in by_menu.items() if len(k) == 2}
    triples = {k: v for k, v in by_menu.items() if len(k) == 3}

    for triple_menu, triple_rec in triples.items():
        if triple_rec["choice"] is None:
            continue
        triple_choice = triple_rec["choice"]

        # For each pairwise subset of this triple
        for i in range(3):
            pair = tuple(sorted([triple_menu[j] for j in range(3) if j != i]))
            if pair not in pairwise:
                continue
            pair_rec = pairwise[pair]
            if pair_rec["choice"] is None:
                continue
            pair_choice = pair_rec["choice"]

            # Check: is the winner of the pair also preferred in the triple?
            # If pair says A>B but triple chooses B (or the third item), that's an IIA violation
            if pair_choice in triple_menu and triple_choice in pair:
                # Both the pair winner and triple winner are in the pair subset
                if pair_choice != triple_choice:
                    violations.append({
                        "pair_menu": list(pair),
                        "triple_menu": list(triple_menu),
                        "pair_choice": pair_choice,
                        "triple_choice": triple_choice,
                        "added_item": triple_menu[i],
                    })

    return violations


def analyze_scenario(scenario_name: str) -> dict:
    """Per-vignette SARP + IIA analysis for one scenario."""
    scenario = ALL_SCENARIOS[scenario_name]
    n_items = len(scenario.items)

    # Load stage1 data
    path = RESPONSE_DIR / f"{scenario_name}__stage1.jsonl"
    if not path.exists():
        print(f"  {scenario_name}: no stage1 data")
        return {}

    records = [json.loads(line) for line in open(path)]
    valid = [r for r in records if r["choice"] is not None]

    # Group by (vignette_id, prompt_name)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        key = f"{r['vignette_id']}__{r['prompt_name']}"
        groups[key].append(r)

    results = {}
    for key, recs in sorted(groups.items()):
        menus = [frozenset(r["menu"]) for r in recs]
        choices = [r["choice"] for r in recs]

        if len(recs) < 5:
            continue

        # SARP test
        log = MenuChoiceLog(menus=menus, choices=choices)
        sarp = validate_menu_sarp(log)
        hm = compute_menu_efficiency(log)

        # IIA violations
        iia = detect_iia_violations(recs, n_items)

        parts = key.split("__")
        vig_id = parts[0]
        prompt = parts[1] if len(parts) > 1 else "?"
        tier = recs[0].get("tier", "?")

        sarp_str = "PASS" if sarp.is_consistent else "FAIL"
        print(f"    {key:<50s} SARP={sarp_str}  HM={hm.efficiency_index:.2f}  "
              f"IIA={len(iia)}  n={len(recs)}")

        results[key] = {
            "vignette_id": vig_id,
            "prompt_name": prompt,
            "tier": tier,
            "n_observations": len(recs),
            "is_sarp": sarp.is_consistent,
            "n_sarp_violations": len(sarp.violations),
            "hm_efficiency": round(hm.efficiency_index, 4),
            "n_iia_violations": len(iia),
            "iia_details": iia,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v2 benchmark results")
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    scenarios = list(ALL_SCENARIOS) if args.all else ([args.scenario] if args.scenario else [])
    if not scenarios:
        parser.print_help()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for sc in scenarios:
        print(f"\n  Analyzing: {ALL_SCENARIOS[sc].display_name}")
        all_results[sc] = analyze_scenario(sc)

    # Save
    summary_path = RESULTS_DIR / "summary_v2.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {summary_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(" v2 SUMMARY: Per-vignette SARP pass rates")
    print("=" * 70)

    for sc in scenarios:
        sresults = all_results.get(sc, {})
        if not sresults:
            continue
        # Group by prompt
        by_prompt: dict[str, list[bool]] = defaultdict(list)
        by_tier: dict[str, list[bool]] = defaultdict(list)
        iia_total = 0

        for data in sresults.values():
            by_prompt[data["prompt_name"]].append(data["is_sarp"])
            by_tier[data["tier"]].append(data["is_sarp"])
            iia_total += data["n_iia_violations"]

        print(f"\n  {sc}:")
        print(f"    By prompt:")
        for prompt in sorted(by_prompt):
            passes = sum(by_prompt[prompt])
            total = len(by_prompt[prompt])
            print(f"      {prompt:<20s} {passes}/{total} SARP-consistent ({passes/total*100:.0f}%)")

        print(f"    By tier:")
        for tier in ["clear", "binary", "ambiguous", "adversarial"]:
            if tier in by_tier:
                passes = sum(by_tier[tier])
                total = len(by_tier[tier])
                print(f"      {tier:<20s} {passes}/{total} ({passes/total*100:.0f}%)")

        print(f"    Total IIA violations: {iia_total}")


if __name__ == "__main__":
    main()
