#!/usr/bin/env python3
"""Stage 3: Analyze benchmark results using PyRevealed.

Builds MenuChoiceLog per (scenario, prompt, model) group and runs:
  1. Engine.analyze_menus() for batch SARP/WARP/HM scoring
  2. Permutation test for SARP violations (H0: uniform random choice)
  3. Bootstrap CIs for HM efficiency
  4. Benjamini-Hochberg FDR correction across all tests

Usage:
    python -m llm_benchmark.analyze --all
    python -m llm_benchmark.analyze --scenario support_ticket
    python -m llm_benchmark.analyze --no-inference   # skip permutation/bootstrap
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyrevealed import MenuChoiceLog
from pyrevealed.algorithms.abstract_choice import validate_menu_sarp, compute_menu_efficiency
from pyrevealed.engine import Engine

from .config import ALL_SCENARIOS, MODEL_CONFIGS, ScenarioConfig

RESPONSE_DIR = Path(__file__).parent / "data" / "responses"
RESULTS_DIR = Path(__file__).parent / "data" / "results"


# =============================================================================
# Statistical Inference
# =============================================================================

def permutation_test_sarp(
    menus: list[list[int]],
    choices: list[int],
    n_items: int,
    observed_violations: int,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test: are SARP violations more than expected by chance?

    H0: Choices are uniformly random from the menu (no preference structure).
    Test statistic: number of SARP violations.
    p-value: fraction of permutations with >= observed violations.
    """
    rng = np.random.default_rng(seed)
    count_geq = 0

    for _ in range(n_permutations):
        # Generate random choices: uniform from each menu
        random_choices = [int(rng.choice(m)) for m in menus]
        log = MenuChoiceLog(
            menus=[frozenset(m) for m in menus],
            choices=random_choices,
        )
        result = validate_menu_sarp(log)
        if len(result.violations) >= observed_violations:
            count_geq += 1

    p_value = count_geq / n_permutations
    return {
        "p_value": round(p_value, 4),
        "n_permutations": n_permutations,
        "observed_violations": observed_violations,
        "mean_random_violations": None,  # could track but expensive
    }


def bootstrap_hm_ci(
    menus: list[list[int]],
    choices: list[int],
    n_items: int,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for HM efficiency.

    Resamples observations with replacement, recomputes HM each time.
    """
    rng = np.random.default_rng(seed)
    n = len(menus)
    hm_values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_menus = [menus[i] for i in indices]
        boot_choices = [choices[i] for i in indices]
        log = MenuChoiceLog(
            menus=[frozenset(m) for m in boot_menus],
            choices=boot_choices,
        )
        result = compute_menu_efficiency(log)
        hm_values.append(result.efficiency_index)

    hm_arr = np.array(hm_values)
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(hm_arr, 100 * alpha / 2))
    ci_upper = float(np.percentile(hm_arr, 100 * (1 - alpha / 2)))

    return {
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "mean_hm": round(float(np.mean(hm_arr)), 4),
        "std_hm": round(float(np.std(hm_arr)), 4),
    }


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg FDR correction.

    Returns list of dicts with original p-value, adjusted p-value, and significance.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    # BH adjustment: p_adj[i] = min(p[i] * n / rank, 1.0), enforcing monotonicity
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adj = min(p * n / rank, prev)
        adj = min(adj, 1.0)
        adjusted[orig_idx] = adj
        prev = adj

    return [
        {
            "p_value": round(p_values[i], 4),
            "p_adjusted": round(adjusted[i], 4),
            "significant": adjusted[i] < alpha,
        }
        for i in range(n)
    ]


# =============================================================================
# Core Analysis
# =============================================================================

def load_responses(scenario_name: str) -> list[dict]:
    """Load all response records for a scenario across all models."""
    records = []
    for model_config in MODEL_CONFIGS:
        path = RESPONSE_DIR / f"{scenario_name}__{model_config['slug']}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def analyze_scenario(
    scenario: ScenarioConfig,
    run_inference: bool = True,
    n_permutations: int = 500,
    n_bootstrap: int = 500,
) -> dict:
    """Analyze all (prompt, model) groups for a single scenario."""
    records = load_responses(scenario.name)
    if not records:
        print(f"  {scenario.name}: no response data found")
        return {}

    # Group by (prompt_name, model)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["choice"] is None:
            continue
        key = f"{r['prompt_name']}__{r['model']}"
        groups[key] = groups.get(key, [])
        groups[key].append(r)

    if not groups:
        print(f"  {scenario.name}: no valid responses")
        return {}

    n_items = len(scenario.items)
    engine_tuples = []
    group_keys = []

    for key in sorted(groups):
        recs = groups[key]
        menus = [r["menu"] for r in recs]
        choices = [r["choice"] for r in recs]
        engine_tuples.append((menus, choices, n_items))
        group_keys.append(key)

    # Batch Engine analysis
    engine = Engine()
    results = engine.analyze_menus(engine_tuples)

    # Build output with optional inference
    scenario_results = {}
    for idx, (key, result) in enumerate(zip(group_keys, results)):
        recs = groups[key]
        menus = [r["menu"] for r in recs]
        choices = [r["choice"] for r in recs]
        hm_eff = result.hm_consistent / result.hm_total if result.hm_total > 0 else 1.0

        # Choice distribution
        dist: dict[str, int] = defaultdict(int)
        for r in recs:
            dist[r["choice_name"]] += 1

        entry: dict = {
            "n_observations": len(recs),
            "is_sarp": result.is_sarp,
            "is_warp": result.is_warp,
            "n_sarp_violations": result.n_sarp_violations,
            "n_warp_violations": result.n_warp_violations,
            "hm_consistent": result.hm_consistent,
            "hm_total": result.hm_total,
            "hm_efficiency": round(hm_eff, 4),
            "max_scc": result.max_scc,
            "choice_distribution": dict(dist),
        }

        # Statistical inference
        if run_inference and len(recs) >= 5:
            perm = permutation_test_sarp(
                menus, choices, n_items,
                result.n_sarp_violations, n_permutations,
            )
            entry["permutation_test"] = perm

            boot = bootstrap_hm_ci(menus, choices, n_items, n_bootstrap)
            entry["bootstrap_hm_ci"] = boot

        sarp_str = "PASS" if result.is_sarp else "FAIL"
        p_str = ""
        if "permutation_test" in entry:
            p = entry["permutation_test"]["p_value"]
            p_str = f"  p={p:.3f}"
        ci_str = ""
        if "bootstrap_hm_ci" in entry:
            ci = entry["bootstrap_hm_ci"]
            ci_str = f"  CI=[{ci['ci_lower']:.3f},{ci['ci_upper']:.3f}]"

        print(f"    {key:<35s} SARP={sarp_str}  HM={hm_eff:.3f}{p_str}{ci_str}  "
              f"violations={result.n_sarp_violations}  n={len(recs)}")

        scenario_results[key] = entry

    return scenario_results


def apply_fdr_correction(all_results: dict) -> dict:
    """Apply Benjamini-Hochberg FDR correction across all permutation tests."""
    # Collect all p-values with their keys
    keys = []
    p_values = []
    for sname, sresults in all_results.items():
        for key, data in sresults.items():
            if "permutation_test" in data:
                keys.append((sname, key))
                p_values.append(data["permutation_test"]["p_value"])

    if not p_values:
        return all_results

    # BH correction
    corrections = benjamini_hochberg(p_values)

    # Write back
    for (sname, key), correction in zip(keys, corrections):
        all_results[sname][key]["permutation_test"]["p_adjusted"] = correction["p_adjusted"]
        all_results[sname][key]["permutation_test"]["bh_significant"] = correction["significant"]

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--scenario", type=str, help="Single scenario name")
    parser.add_argument("--all", action="store_true", help="Analyze all scenarios")
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip permutation tests and bootstrap CIs")
    parser.add_argument("--permutations", type=int, default=500,
                        help="Number of permutations for SARP test")
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="Number of bootstrap samples for HM CI")
    args = parser.parse_args()

    if args.all:
        scenarios = list(ALL_SCENARIOS.values())
    elif args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            return
        scenarios = [ALL_SCENARIOS[args.scenario]]
    else:
        parser.print_help()
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_inference = not args.no_inference

    all_results = {}
    for scenario in scenarios:
        print(f"\n  Analyzing: {scenario.display_name}")
        all_results[scenario.name] = analyze_scenario(
            scenario, run_inference, args.permutations, args.bootstrap,
        )

    # FDR correction across all tests
    if run_inference:
        all_results = apply_fdr_correction(all_results)
        print("\n  Applied Benjamini-Hochberg FDR correction across all tests")

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print cross-scenario comparison
    print("\n" + "=" * 70)
    print(" CROSS-SCENARIO SUMMARY")
    print("=" * 70)

    header = (f"  {'Scenario':<20s} {'Model':<14s} {'Prompt':<16s} "
              f"{'SARP':>5s} {'HM':>6s} {'Viol':>5s}")
    if run_inference:
        header += f" {'p-val':>6s} {'p-adj':>6s} {'Sig':>4s} {'HM 95% CI':>16s}"
    print(f"\n{header}")
    print(f"  {'-'*20} {'-'*14} {'-'*16} {'-'*5} {'-'*6} {'-'*5}", end="")
    if run_inference:
        print(f" {'-'*6} {'-'*6} {'-'*4} {'-'*16}", end="")
    print()

    for sname, sresults in all_results.items():
        for key, data in sorted(sresults.items()):
            parts = key.split("__")
            prompt_name = parts[0]
            model = parts[1] if len(parts) > 1 else "?"
            sarp = "PASS" if data["is_sarp"] else "FAIL"
            line = (f"  {sname:<20s} {model:<14s} {prompt_name:<16s} "
                    f"{sarp:>5s} {data['hm_efficiency']:>6.3f} "
                    f"{data['n_sarp_violations']:>5d}")
            if run_inference and "permutation_test" in data:
                pt = data["permutation_test"]
                ci = data.get("bootstrap_hm_ci", {})
                sig = "*" if pt.get("bh_significant", False) else ""
                ci_str = f"[{ci.get('ci_lower', 0):.3f},{ci.get('ci_upper', 1):.3f}]"
                line += (f" {pt['p_value']:>6.3f} {pt.get('p_adjusted', pt['p_value']):>6.3f}"
                         f" {sig:>4s} {ci_str:>16s}")
            print(line)


if __name__ == "__main__":
    main()
