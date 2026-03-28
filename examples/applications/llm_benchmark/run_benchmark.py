#!/usr/bin/env python3
"""Stage 2: Run the LLM benchmark - query models with vignettes and prompts.

Append-only JSONL: each run adds missing (prompt, trial) pairs. Safe to
interrupt and resume. Data accumulates across runs.

Usage:
    python -m llm_benchmark.run_benchmark --scenario support_ticket --model gpt-4o-mini --trials 20
    python -m llm_benchmark.run_benchmark --all --trials 200
    python -m llm_benchmark.run_benchmark --all --trials 20   # small first pass
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ALL_SCENARIOS, MODEL_CONFIGS, ScenarioConfig

VIGNETTE_DIR = Path(__file__).parent / "data" / "vignettes"
RESPONSE_DIR = Path(__file__).parent / "data" / "responses"


def load_vignettes(scenario_name: str, n_trials: int) -> list[dict]:
    """Load pre-generated vignettes for a scenario."""
    path = VIGNETTE_DIR / f"{scenario_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No vignettes at {path}. Run generate_vignettes.py first."
        )
    vignettes = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if record["trial"] < n_trials:
                vignettes.append(record)
    vignettes.sort(key=lambda r: r["trial"])
    return vignettes


def load_existing_responses(path: Path) -> set[tuple[str, int]]:
    """Load existing (prompt_name, trial) pairs from response JSONL."""
    done: set[tuple[str, int]] = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["prompt_name"], r["trial"]))
    return done


def format_user_prompt(
    scenario: ScenarioConfig,
    vignette: str,
    menu: list[int],
) -> str:
    """Build the user prompt from scenario context + vignette + menu items."""
    sorted_menu = sorted(menu)
    # Use letter labels for unambiguous parsing
    labels = "ABCDE"
    options = "\n".join(
        f"({labels[j]}) {scenario.items[i]}: {scenario.item_descriptions[i]}"
        for j, i in enumerate(sorted_menu)
    )
    valid_letters = ", ".join(labels[j] for j in range(len(sorted_menu)))
    return (
        f"{scenario.context}\n\n"
        f"Situation:\n{vignette}\n\n"
        f"Choose from ONLY these options:\n{options}\n\n"
        f"Reply with ONLY the letter ({valid_letters}). Do not explain."
    )


def parse_choice(response: str, menu: list[int], items: dict[int, str]) -> int | None:
    """Parse LLM response to extract the chosen action.

    Tries: (1) letter label match, (2) exact item name match, (3) substring match.
    """
    sorted_menu = sorted(menu)
    labels = "ABCDE"
    response_clean = response.strip().upper()

    # Try letter match: "A", "(A)", "A)", "A."
    for j, idx in enumerate(sorted_menu):
        letter = labels[j]
        if response_clean in (letter, f"({letter})", f"{letter})", f"{letter}."):
            return idx

    # Try letter anywhere in short response
    if len(response_clean) <= 5:
        for j, idx in enumerate(sorted_menu):
            if labels[j] in response_clean:
                return idx

    # Fallback: try item name match
    response_lower = response.strip().lower()
    for idx in sorted_menu:
        if items[idx].lower() == response_lower:
            return idx
    for idx in sorted_menu:
        if items[idx].lower() in response_lower:
            return idx

    return None


def query_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
) -> str:
    """Query OpenAI API. Handles both chat and reasoning models."""
    from openai import OpenAI

    client = OpenAI()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs: dict = {"model": model_name, "messages": messages}
    if model_name.startswith("o"):
        # Reasoning models use internal CoT tokens; need generous budget
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["max_tokens"] = 50
        if temperature is not None:
            kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def run_scenario_model(
    scenario: ScenarioConfig,
    model_config: dict,
    n_trials: int,
    delay: float = 0.1,
) -> int:
    """Run benchmark for one (scenario, model) pair. Returns number of new records."""
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    model_name = model_config["name"]
    model_slug = model_config["slug"]
    temperature = model_config.get("temperature")
    out_path = RESPONSE_DIR / f"{scenario.name}__{model_slug}.jsonl"

    vignettes = load_vignettes(scenario.name, n_trials)
    if len(vignettes) < n_trials:
        print(f"    WARNING: only {len(vignettes)} vignettes available (requested {n_trials})")
        n_trials = len(vignettes)

    done = load_existing_responses(out_path)
    prompt_names = list(scenario.system_prompts.keys())

    # Build list of missing (prompt, trial) pairs
    missing = []
    for pname in prompt_names:
        for v in vignettes:
            if (pname, v["trial"]) not in done:
                missing.append((pname, v))

    if not missing:
        print(f"    {scenario.name} x {model_name}: all done ({len(done)} records)")
        return 0

    print(f"    {scenario.name} x {model_name}: {len(missing)} calls needed "
          f"({len(done)} cached)")

    new_count = 0
    with open(out_path, "a") as f:
        for i, (pname, vignette) in enumerate(missing):
            system_prompt = scenario.system_prompts[pname]
            user_prompt = format_user_prompt(scenario, vignette["vignette"], vignette["menu"])

            try:
                response_text = query_llm(model_name, system_prompt, user_prompt, temperature)
                choice = parse_choice(response_text, vignette["menu"], scenario.items)
            except Exception as e:
                response_text = f"ERROR: {e}"
                choice = None

            record = {
                "scenario": scenario.name,
                "prompt_name": pname,
                "model": model_name,
                "trial": vignette["trial"],
                "menu": vignette["menu"],
                "vignette": vignette["vignette"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": response_text,
                "choice": choice,
                "choice_name": scenario.items.get(choice, "PARSE_FAIL") if choice is not None else "PARSE_FAIL",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            new_count += 1

            if (i + 1) % 50 == 0:
                valid = new_count  # rough estimate
                print(f"      progress: {i + 1}/{len(missing)} calls "
                      f"({len(done) + new_count} total)")

            if delay > 0:
                time.sleep(delay)

    print(f"    {scenario.name} x {model_name}: +{new_count} new records "
          f"({len(done) + new_count} total)")
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM benchmark")
    parser.add_argument("--scenario", type=str, help="Single scenario name")
    parser.add_argument("--model", type=str, help="Single model name (e.g., gpt-4o-mini)")
    parser.add_argument("--all", action="store_true", help="Run all scenarios and models")
    parser.add_argument("--trials", type=int, default=200, help="Trials per prompt")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between API calls")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Determine scenarios
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

    # Determine models
    if args.model:
        model_configs = [m for m in MODEL_CONFIGS if m["name"] == args.model]
        if not model_configs:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(m['name'] for m in MODEL_CONFIGS)}")
            return
    else:
        model_configs = MODEL_CONFIGS

    total_new = 0
    total_calls = len(scenarios) * len(model_configs) * len(list(ALL_SCENARIOS.values())[0].system_prompts) * args.trials
    print(f"Benchmark: {len(scenarios)} scenarios x {len(model_configs)} models x "
          f"5 prompts x {args.trials} trials = up to {total_calls} calls\n")

    for scenario in scenarios:
        for model_config in model_configs:
            new = run_scenario_model(scenario, model_config, args.trials, args.delay)
            total_new += new

    print(f"\nDone. {total_new} new API calls made.")


if __name__ == "__main__":
    main()
