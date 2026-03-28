"""Stochastic choice EDA: does StochasticChoiceLog add value over MenuChoiceLog?

Key empirical question: do exact menus repeat often enough per user to make
stochastic RP (RUM / IIA / Regularity) meaningful? If repeat_rate ≈ 0 every
menu is seen once and RUM tests have no discriminating power.

This script uses synthetic data that faithfully mimics:
  - REES46:        large product catalog, session-based, menus rarely repeat
  - Taobao (6h):   smaller per-user item set, buy-anchored, menus repeat slightly more

Swap the synthetic generators for real loaders when Kaggle data is available.

Run:
    python3 case_studies/eda/stochastic_menu_eda.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prefgraph.algorithms.abstract_choice import (
    compute_menu_efficiency,
    validate_menu_sarp,
)
from prefgraph.contrib.stochastic import (
    check_independence_irrelevant_alternatives,
    test_regularity,
    test_rum_consistency,
    test_stochastic_transitivity,
)
from prefgraph.core.session import MenuChoiceLog, StochasticChoiceLog


# --------------------------------------------------------------------------- #
# Synthetic data generators (seed=42, deterministic)
# --------------------------------------------------------------------------- #

RNG = np.random.default_rng(42)


def _synthetic_rees46_user(uid: int) -> MenuChoiceLog:
    """Mimic REES46 structure: large catalog (~300 items/user), 15 sessions.

    Each session draws 3-10 items from the catalog. With 300 items and
    small menus the probability of exact menu overlap is near zero —
    exactly what we observe in the real data.
    """
    catalog_size = 300
    n_sessions = 15
    menus, choices = [], []
    for _ in range(n_sessions):
        size = int(RNG.integers(3, 11))
        menu = frozenset(RNG.choice(catalog_size, size=size, replace=False).tolist())
        choice = int(RNG.choice(list(menu)))
        menus.append(menu)
        choices.append(choice)
    return MenuChoiceLog(menus=menus, choices=choices, user_id=f"rees46_{uid}")


def _synthetic_taobao_user(uid: int) -> MenuChoiceLog:
    """Mimic Taobao buy-anchored structure: smaller per-user item set (~40 items),
    20 observations. Smaller item set means more chance of menu repeats, but
    still low because 6h windows draw different subsets each time.
    """
    catalog_size = 40
    n_obs = 20
    menus, choices = [], []
    for _ in range(n_obs):
        size = int(RNG.integers(2, 8))
        menu = frozenset(RNG.choice(catalog_size, size=size, replace=False).tolist())
        choice = int(RNG.choice(list(menu)))
        menus.append(menu)
        choices.append(choice)
    return MenuChoiceLog(menus=menus, choices=choices, user_id=f"taobao_{uid}")


def _synthetic_repeat_user(uid: int) -> MenuChoiceLog:
    """Synthetic user where menus DO repeat — to show what stochastic RP looks
    like when it actually has data. Tiny catalog (8 items), 30 observations,
    menus of size 2-4. High repeat rate expected (~50%+).
    """
    catalog_size = 8
    n_obs = 30
    menus, choices = [], []
    for _ in range(n_obs):
        size = int(RNG.integers(2, 5))
        menu = frozenset(RNG.choice(catalog_size, size=size, replace=False).tolist())
        choice = int(RNG.choice(list(menu)))
        menus.append(menu)
        choices.append(choice)
    return MenuChoiceLog(menus=menus, choices=choices, user_id=f"repeat_{uid}")


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #

def _menu_repeat_stats(log: MenuChoiceLog) -> dict:
    """Polars-based menu repeat diagnostics.

    repeat_rate = 1 - distinct_menus / n_obs
    Near zero → stochastic RP has no power (each menu seen once).
    """
    n_obs = len(log.choices)
    keys = pl.Series("menu", [str(sorted(m)) for m in log.menus])
    counts = keys.value_counts().rename({"count": "freq"})
    n_distinct = len(counts)
    repeat_rate = 1.0 - n_distinct / n_obs if n_obs else 0.0
    max_freq = int(counts["freq"].max())
    hist = (
        counts.group_by("freq").agg(pl.len().alias("n")).sort("freq")
    )
    hist_str = "  ".join(f"{r['freq']}x:{r['n']}" for r in hist.iter_rows(named=True))
    return {
        "n_obs": n_obs,
        "n_distinct": n_distinct,
        "repeat_rate": round(repeat_rate, 3),
        "max_freq": max_freq,
        "hist": hist_str,
    }


def _det_stats(log: MenuChoiceLog) -> dict:
    sizes = pl.Series("sz", [len(m) for m in log.menus])
    sarp = validate_menu_sarp(log)
    hm = compute_menu_efficiency(log)
    return {
        "n_items": log.num_items,
        "mean_sz": round(float(sizes.mean()), 1),
        "max_sz": int(sizes.max()),
        "sarp_ok": sarp.is_consistent,
        "hm": round(hm.efficiency_index, 3),
    }


def _stoch_rp(stoch: StochasticChoiceLog) -> dict:
    """Stochastic RP tests — only meaningful when repeat_rate >= 0.10."""
    rum = test_rum_consistency(stoch)
    reg = test_regularity(stoch)
    trans = test_stochastic_transitivity(stoch)
    iia = check_independence_irrelevant_alternatives(stoch)
    return {
        "rum": rum.is_rum_consistent,
        "d_rum": round(rum.distance_to_rum, 4),
        "reg": reg.satisfies_regularity,
        "reg_v": len(reg.violations),
        "wst": trans.satisfies_wst,
        "triples": trans.num_testable_triples,
        "iia": iia,
    }


def analyze(dataset: str, log: MenuChoiceLog) -> dict:
    uid = log.user_id or dataset
    det = _det_stats(log)
    rep = _menu_repeat_stats(log)

    print(f"\n  [{dataset}]  user={uid}")
    print(f"    obs={rep['n_obs']}  items={det['n_items']}  "
          f"menu_sz=mean{det['mean_sz']}/max{det['max_sz']}")
    print(f"    SARP={'OK' if det['sarp_ok'] else 'VIOLATIONS'}  HM={det['hm']}")
    print(f"    distinct={rep['n_distinct']}/{rep['n_obs']}  "
          f"repeat_rate={rep['repeat_rate']}  max_freq={rep['max_freq']}")
    print(f"    freq_hist: {rep['hist']}")

    stoch = StochasticChoiceLog.from_repeated_choices(
        menus=log.menus, choices=log.choices
    )

    s: dict = {}
    if rep["repeat_rate"] >= 0.10:
        print("    → sufficient repeats — running stochastic RP")
        s = _stoch_rp(stoch)
        print(f"    RUM={s['rum']} dist={s['d_rum']}  "
              f"Reg={s['reg']}({s['reg_v']}v)  WST={s['wst']}  "
              f"triples={s['triples']}  IIA={s['iia']}")
    else:
        print("    → repeat_rate < 0.10 — stochastic RP skipped (no power)")

    return {"dataset": dataset, "user": uid, **det, **rep, **s}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rows: list[dict] = []

    print("=" * 65)
    print("REES46-like  (large catalog, sessions rarely repeat)")
    print("=" * 65)
    for i in range(3):
        rows.append(analyze("REES46", _synthetic_rees46_user(i)))

    print("\n" + "=" * 65)
    print("Taobao-like  (smaller catalog, buy-anchored 6h window)")
    print("=" * 65)
    for i in range(3):
        rows.append(analyze("Taobao", _synthetic_taobao_user(i)))

    print("\n" + "=" * 65)
    print("Repeat-heavy  (tiny catalog — shows stochastic RP in action)")
    print("=" * 65)
    for i in range(2):
        rows.append(analyze("RepeatHeavy", _synthetic_repeat_user(i)))

    # --- Summary table via Polars ---
    df = pl.DataFrame(rows, infer_schema_length=len(rows))
    stoch_cols = ["rum", "d_rum", "reg", "wst", "iia"]
    for c in stoch_cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias(c))

    display = ["dataset", "user", "n_obs", "n_distinct", "repeat_rate",
               "sarp_ok", "hm", "rum", "d_rum", "reg", "wst", "iia"]
    print("\n\n" + "=" * 95)
    print("SUMMARY")
    print("repeat_rate < 0.10 → stochastic tests have no discriminating power")
    print("=" * 95)
    print(df.select(display))
    print("=" * 95)
    print()
    print("NOTE: swap _synthetic_*_user() for load_taobao()/load_rees46() for real data.")
    print("      Expect repeat_rate ≈ 0.00 on both real datasets (large catalogs,")
    print("      session-specific views) → MenuChoiceLog (deterministic) is the right frame.")
