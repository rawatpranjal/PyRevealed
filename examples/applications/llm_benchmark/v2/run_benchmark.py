#!/usr/bin/env python3
"""v2 Stage 2: Run per-vignette benchmark.

For each (vignette, menu, prompt): query the LLM.
Stage 1 (deterministic): temp=0, 1 response per menu.
Stage 2 (stochastic): temp=0.7, K responses per menu (on Stage 1 failures only).

Usage:
    python -m applications.llm_benchmark.v2.run_benchmark --all --stage 1
    python -m applications.llm_benchmark.v2.run_benchmark --all --stage 2 --k 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import ALL_SCENARIOS
from ..run_benchmark import format_user_prompt, parse_choice, query_llm
from .config import MODEL, STOCHASTIC_TEMPERATURE, STOCHASTIC_K

VIGNETTE_DIR = Path(__file__).parent / "data" / "vignettes"
RESPONSE_DIR = Path(__file__).parent / "data" / "responses"


def load_vignettes(scenario_name: str) -> list[dict]:
    path = VIGNETTE_DIR / f"{scenario_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No vignettes at {path}")
    return [json.loads(line) for line in open(path)]


def load_existing(path: Path) -> set[tuple[str, str, int, int]]:
    """Load existing (vignette_id, prompt_name, menu_idx, rep) keys."""
    done: set[tuple[str, str, int, int]] = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["vignette_id"], r["prompt_name"], r["menu_idx"], r.get("rep", 0)))
    return done


def run_stage1(scenario_name: str) -> int:
    """Deterministic screen: temp=0, 1 response per (vignette, menu, prompt)."""
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    scenario = ALL_SCENARIOS[scenario_name]
    vignettes = load_vignettes(scenario_name)
    out_path = RESPONSE_DIR / f"{scenario_name}__stage1.jsonl"
    done = load_existing(out_path)

    prompts = scenario.system_prompts
    new_count = 0

    with open(out_path, "a") as f:
        for vig in vignettes:
            for prompt_name, system_prompt in prompts.items():
                for menu_idx, menu in enumerate(vig["menus"]):
                    key = (vig["vignette_id"], prompt_name, menu_idx, 0)
                    if key in done:
                        continue

                    user_prompt = format_user_prompt(scenario, vig["vignette"], menu)
                    try:
                        response_text = query_llm(
                            MODEL["name"], system_prompt, user_prompt, MODEL["temperature"]
                        )
                        choice = parse_choice(response_text, menu, scenario.items)
                    except Exception as e:
                        response_text = f"ERROR: {e}"
                        choice = None

                    record = {
                        "scenario": scenario_name,
                        "vignette_id": vig["vignette_id"],
                        "vignette_idx": vig["vignette_idx"],
                        "tier": vig["tier"],
                        "prompt_name": prompt_name,
                        "menu_idx": menu_idx,
                        "menu": menu,
                        "menu_size": len(menu),
                        "rep": 0,
                        "temperature": MODEL["temperature"],
                        "vignette": vig["vignette"],
                        "response": response_text,
                        "choice": choice,
                        "choice_name": scenario.items.get(choice, "PARSE_FAIL") if choice is not None else "PARSE_FAIL",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    done.add(key)
                    new_count += 1

                    if new_count % 100 == 0:
                        print(f"      {new_count} calls...")
                    time.sleep(0.05)

    total = len(done)
    print(f"    {scenario_name} stage1: +{new_count} new ({total} total)")
    return new_count


def run_stage2(scenario_name: str, k: int = STOCHASTIC_K) -> int:
    """Stochastic deep dive: temp=0.7, K reps on Stage 1 SARP failures."""
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    scenario = ALL_SCENARIOS[scenario_name]
    vignettes = load_vignettes(scenario_name)

    # Load stage1 results to identify failures
    stage1_path = RESPONSE_DIR / f"{scenario_name}__stage1.jsonl"
    if not stage1_path.exists():
        print(f"    {scenario_name}: no stage1 data, run stage1 first")
        return 0

    # Identify SARP failures per (vignette, prompt) - will be done by analyze
    # For now, run stochastic on ALL vignettes (can filter later)
    out_path = RESPONSE_DIR / f"{scenario_name}__stage2.jsonl"
    done = load_existing(out_path)

    prompts = scenario.system_prompts
    new_count = 0

    with open(out_path, "a") as f:
        for vig in vignettes:
            for prompt_name, system_prompt in prompts.items():
                for menu_idx, menu in enumerate(vig["menus"]):
                    for rep in range(k):
                        key = (vig["vignette_id"], prompt_name, menu_idx, rep)
                        if key in done:
                            continue

                        user_prompt = format_user_prompt(scenario, vig["vignette"], menu)
                        try:
                            response_text = query_llm(
                                MODEL["name"], system_prompt, user_prompt,
                                STOCHASTIC_TEMPERATURE,
                            )
                            choice = parse_choice(response_text, menu, scenario.items)
                        except Exception as e:
                            response_text = f"ERROR: {e}"
                            choice = None

                        record = {
                            "scenario": scenario_name,
                            "vignette_id": vig["vignette_id"],
                            "vignette_idx": vig["vignette_idx"],
                            "tier": vig["tier"],
                            "prompt_name": prompt_name,
                            "menu_idx": menu_idx,
                            "menu": menu,
                            "menu_size": len(menu),
                            "rep": rep,
                            "temperature": STOCHASTIC_TEMPERATURE,
                            "vignette": vig["vignette"],
                            "response": response_text,
                            "choice": choice,
                            "choice_name": scenario.items.get(choice, "PARSE_FAIL") if choice is not None else "PARSE_FAIL",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        done.add(key)
                        new_count += 1

                        if new_count % 200 == 0:
                            print(f"      {new_count} calls...")
                        time.sleep(0.05)

    total = len(done)
    print(f"    {scenario_name} stage2: +{new_count} new ({total} total, k={k})")
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 benchmark")
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--k", type=int, default=STOCHASTIC_K, help="Stochastic repetitions")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    scenarios = list(ALL_SCENARIOS) if args.all else ([args.scenario] if args.scenario else [])
    if not scenarios:
        parser.print_help()
        return

    runner = run_stage1 if args.stage == 1 else lambda sc: run_stage2(sc, args.k)
    total = 0
    for sc in scenarios:
        total += runner(sc)
    print(f"\nDone. {total} new calls.")


if __name__ == "__main__":
    main()
