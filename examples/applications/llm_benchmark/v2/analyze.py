#!/usr/bin/env python3
"""v2 Stage 3: Per-vignette SARP + IIA analysis.

For each (vignette, prompt): test SARP on 15 menu choices.
Detect IIA violations by comparing pairwise vs triple menus.

Usage:
    python -m applications.llm_benchmark.v2.analyze --all
    python -m applications.llm_benchmark.v2.analyze --all --stage 2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from prefgraph import MenuChoiceLog
from prefgraph.algorithms.abstract_choice import (
    validate_menu_sarp,
    validate_menu_warp,
    compute_menu_efficiency,
)

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


def analyze_scenario_stage2(scenario_name: str) -> dict:
    """Per-vignette SARP/WARP/HM/IIA analysis on majority-vote stage2 data.

    Also computes per-menu mixing rate and agreement with stage1.
    """
    scenario = ALL_SCENARIOS[scenario_name]
    n_items = len(scenario.items)

    s2_path = RESPONSE_DIR / f"{scenario_name}__stage2.jsonl"
    if not s2_path.exists():
        print(f"  {scenario_name}: no stage2 data")
        return {}

    s2_records = [json.loads(line) for line in open(s2_path)]

    # Load stage1 for agreement comparison
    s1_path = RESPONSE_DIR / f"{scenario_name}__stage1.jsonl"
    s1_by_key: dict[str, int | None] = {}
    if s1_path.exists():
        for line in open(s1_path):
            r = json.loads(line)
            k = f"{r['vignette_id']}__{r['prompt_name']}__{r['menu_idx']}"
            s1_by_key[k] = r["choice"]

    # Group stage2 by (vignette_id, prompt_name, menu_idx) -> list of choices
    menu_groups: dict[str, list] = defaultdict(list)
    meta: dict[str, dict] = {}
    for r in s2_records:
        k = f"{r['vignette_id']}__{r['prompt_name']}__{r['menu_idx']}"
        menu_groups[k].append(r["choice"])
        if k not in meta:
            meta[k] = {
                "vignette_id": r["vignette_id"],
                "prompt_name": r["prompt_name"],
                "menu_idx": r["menu_idx"],
                "menu": r["menu"],
                "tier": r.get("tier", "?"),
            }

    # Compute majority-vote choice and mixing stats per menu
    majority: dict[str, int | None] = {}
    mixed_flags: dict[str, bool] = {}
    for k, choices in menu_groups.items():
        valid = [c for c in choices if c is not None]
        if not valid:
            majority[k] = None
            mixed_flags[k] = False
            continue
        counts = Counter(valid)
        top, top_count = counts.most_common(1)[0]
        # Tie-break: if top two are tied, mark None (ambiguous)
        if len(counts) > 1 and counts.most_common(2)[1][1] == top_count:
            majority[k] = None
        else:
            majority[k] = top
        mixed_flags[k] = len(counts) > 1

    # Group by (vignette_id, prompt_name) for SARP/WARP/HM/IIA
    vp_groups: dict[str, list[str]] = defaultdict(list)
    for k, m in meta.items():
        vp_key = f"{m['vignette_id']}__{m['prompt_name']}"
        vp_groups[vp_key].append(k)

    results = {}
    for vp_key, menu_keys in sorted(vp_groups.items()):
        # Build majority-vote records for this vignette+prompt
        maj_records = []
        n_mixed = 0
        n_agree = 0
        n_agree_total = 0
        for mk in sorted(menu_keys, key=lambda x: meta[x]["menu_idx"]):
            m = meta[mk]
            maj_choice = majority[mk]
            if mixed_flags[mk]:
                n_mixed += 1
            # Agreement with stage1
            s1_key = f"{m['vignette_id']}__{m['prompt_name']}__{m['menu_idx']}"
            if s1_key in s1_by_key and maj_choice is not None:
                n_agree_total += 1
                if s1_by_key[s1_key] == maj_choice:
                    n_agree += 1
            if maj_choice is not None:
                maj_records.append({
                    "menu": m["menu"],
                    "choice": maj_choice,
                })

        n_menus = len(menu_keys)
        if len(maj_records) < 5:
            continue

        menus = [frozenset(r["menu"]) for r in maj_records]
        choices = [r["choice"] for r in maj_records]

        log = MenuChoiceLog(menus=menus, choices=choices)
        sarp = validate_menu_sarp(log)
        warp = validate_menu_warp(log)
        hm = compute_menu_efficiency(log)
        iia = detect_iia_violations(maj_records, n_items)

        parts = vp_key.split("__")
        vig_id = parts[0]
        prompt = parts[1] if len(parts) > 1 else "?"
        tier = meta[menu_keys[0]]["tier"]

        sarp_str = "PASS" if sarp.is_consistent else "FAIL"
        warp_str = "PASS" if warp.is_consistent else "FAIL"
        pct_mixed = round(n_mixed / n_menus * 100, 1) if n_menus else 0
        pct_agree = round(n_agree / n_agree_total * 100, 1) if n_agree_total else 0

        print(f"    {vp_key:<50s} SARP={sarp_str} WARP={warp_str} HM={hm.efficiency_index:.2f}  "
              f"IIA={len(iia)}  mixed={pct_mixed}%  agree={pct_agree}%  n={len(maj_records)}")

        results[vp_key] = {
            "vignette_id": vig_id,
            "prompt_name": prompt,
            "tier": tier,
            "n_menus": n_menus,
            "n_valid_majority": len(maj_records),
            "is_sarp": sarp.is_consistent,
            "n_sarp_violations": len(sarp.violations),
            "is_warp": warp.is_consistent,
            "n_warp_violations": len(warp.violations),
            "hm_efficiency": round(hm.efficiency_index, 4),
            "n_iia_violations": len(iia),
            "iia_details": iia,
            "pct_mixed": pct_mixed,
            "pct_agreement": pct_agree,
        }

    return results


def print_stage2_summary(all_results: dict, scenarios: list[str]) -> None:
    """Print standout findings from stage2 analysis."""
    print("\n" + "=" * 80)
    print(" v2 STOCHASTIC SUMMARY (majority-vote from K=20, temp=0.7)")
    print("=" * 80)

    for sc in scenarios:
        sresults = all_results.get(sc, {})
        if not sresults:
            continue

        by_prompt_sarp: dict[str, list[bool]] = defaultdict(list)
        by_prompt_warp: dict[str, list[bool]] = defaultdict(list)
        by_tier_sarp: dict[str, list[bool]] = defaultdict(list)
        by_prompt_mixed: dict[str, list[float]] = defaultdict(list)
        by_prompt_agree: dict[str, list[float]] = defaultdict(list)
        by_prompt_hm: dict[str, list[float]] = defaultdict(list)
        iia_total = 0

        for data in sresults.values():
            p = data["prompt_name"]
            by_prompt_sarp[p].append(data["is_sarp"])
            by_prompt_warp[p].append(data["is_warp"])
            by_tier_sarp[data["tier"]].append(data["is_sarp"])
            by_prompt_mixed[p].append(data["pct_mixed"])
            by_prompt_agree[p].append(data["pct_agreement"])
            by_prompt_hm[p].append(data["hm_efficiency"])
            iia_total += data["n_iia_violations"]

        display = ALL_SCENARIOS[sc].display_name
        print(f"\n  {display} ({sc})")
        print(f"  {'─' * 70}")

        # SARP by prompt
        print(f"    SARP pass rate (majority-vote):")
        prompt_order = ["minimal", "decision_tree", "conservative", "aggressive", "chain_of_thought"]
        for prompt in prompt_order:
            if prompt in by_prompt_sarp:
                passes = sum(by_prompt_sarp[prompt])
                total = len(by_prompt_sarp[prompt])
                pct = passes / total * 100
                print(f"      {prompt:<22s} {passes:>2}/{total} = {pct:5.1f}%")

        # SARP by tier
        print(f"    SARP pass rate by tier:")
        for tier in ["clear", "binary", "ambiguous", "adversarial"]:
            if tier in by_tier_sarp:
                passes = sum(by_tier_sarp[tier])
                total = len(by_tier_sarp[tier])
                print(f"      {tier:<22s} {passes:>2}/{total} = {passes/total*100:5.1f}%")

        # WARP by prompt
        print(f"    WARP pass rate:")
        for prompt in prompt_order:
            if prompt in by_prompt_warp:
                passes = sum(by_prompt_warp[prompt])
                total = len(by_prompt_warp[prompt])
                print(f"      {prompt:<22s} {passes:>2}/{total} = {passes/total*100:5.1f}%")

        # HM mean by prompt
        print(f"    Mean HM efficiency:")
        for prompt in prompt_order:
            if prompt in by_prompt_hm:
                mean_hm = sum(by_prompt_hm[prompt]) / len(by_prompt_hm[prompt])
                print(f"      {prompt:<22s} {mean_hm:.3f}")

        # % mixed
        print(f"    Mean % mixed menus:")
        for prompt in prompt_order:
            if prompt in by_prompt_mixed:
                mean_mixed = sum(by_prompt_mixed[prompt]) / len(by_prompt_mixed[prompt])
                print(f"      {prompt:<22s} {mean_mixed:5.1f}%")

        # Agreement
        print(f"    Mean agreement with deterministic (temp=0):")
        for prompt in prompt_order:
            if prompt in by_prompt_agree:
                mean_agree = sum(by_prompt_agree[prompt]) / len(by_prompt_agree[prompt])
                print(f"      {prompt:<22s} {mean_agree:5.1f}%")

        print(f"    Total IIA violations: {iia_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v2 benchmark results")
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    scenarios = list(ALL_SCENARIOS) if args.all else ([args.scenario] if args.scenario else [])
    if not scenarios:
        parser.print_help()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.stage == 2:
        # Filter to scenarios with complete stage2 data (15000 lines)
        ready = []
        for sc in scenarios:
            p = RESPONSE_DIR / f"{sc}__stage2.jsonl"
            if p.exists():
                n = sum(1 for _ in open(p))
                if n >= 15000:
                    ready.append(sc)
                else:
                    print(f"  {sc}: stage2 incomplete ({n}/15000), skipping")
            else:
                print(f"  {sc}: no stage2 data, skipping")
        scenarios = ready

        for sc in scenarios:
            print(f"\n  Analyzing (stage2): {ALL_SCENARIOS[sc].display_name}")
            all_results[sc] = analyze_scenario_stage2(sc)

        summary_path = RESULTS_DIR / "summary_v2_stage2.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {summary_path}")

        print_stage2_summary(all_results, scenarios)
    else:
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
