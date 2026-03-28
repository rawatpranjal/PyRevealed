#!/usr/bin/env python3
"""1M user Menu Choice benchmark - rec/search click data at scale."""

import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from prefgraph.engine import Engine

CPU = os.cpu_count() or 1
N = 1_000_000
CHUNK = 50_000
N_ITEMS = 50
SESSIONS_LO, SESSIONS_HI = 20, 100
MENU_LO, MENU_HI = 5, 20


def generate_menu_chunk(n_users, seed):
    """Fast menu data generation using numpy."""
    rng = np.random.default_rng(seed)
    users = []
    for _ in range(n_users):
        n_sess = rng.integers(SESSIONS_LO, SESSIONS_HI + 1)
        menus = []
        choices = []
        # Pre-generate all menu sizes and items
        sizes = rng.integers(MENU_LO, MENU_HI + 1, size=n_sess)
        for s in range(n_sess):
            sz = int(sizes[s])
            menu = rng.choice(N_ITEMS, size=sz, replace=False).tolist()
            menu.sort()
            choices.append(menu[rng.integers(0, sz)])
            menus.append(menu)
        users.append((menus, choices, N_ITEMS))
    return users


def fmt(s):
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    return f"{s / 60:.1f}min"


def main():
    print("=" * 70)
    print(f" MENU CHOICE: {N:,} User Benchmark ({CPU} cores)")
    print("=" * 70)
    print(f" Sessions/user: {SESSIONS_LO}-{SESSIONS_HI}")
    print(f" Items:         {N_ITEMS}")
    print(f" Metrics:       SARP + WARP + Houtman-Maks")
    print(f" Chunks:        {CHUNK:,}")
    print("=" * 70)

    engine = Engine(chunk_size=CHUNK)

    # Warmup
    engine.analyze_menus(generate_menu_chunk(20, seed=0))

    tracemalloc.start()
    t_start = time.perf_counter()

    all_results = []
    processed = 0
    chunk_idx = 0

    while processed < N:
        batch_size = min(CHUNK, N - processed)

        t_gen = time.perf_counter()
        chunk = generate_menu_chunk(batch_size, seed=processed)
        gen_time = time.perf_counter() - t_gen

        t_analyze = time.perf_counter()
        results = engine.analyze_menus(chunk)
        analyze_time = time.perf_counter() - t_analyze

        all_results.extend(results)
        processed += batch_size
        chunk_idx += 1
        _, peak = tracemalloc.get_traced_memory()

        print(f"  Chunk {chunk_idx:3d}/{N // CHUNK}: "
              f"gen={gen_time:.1f}s  analyze={analyze_time:.1f}s  "
              f"mem={peak / 1e6:.0f}MB  ({processed:,}/{N:,})")

        del chunk, results

    total_time = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Results
    n_sarp = sum(1 for r in all_results if r.is_sarp)
    n_warp = sum(1 for r in all_results if r.is_warp)
    hm = np.array([r.hm_consistent / max(r.hm_total, 1) for r in all_results])
    times = np.array([r.compute_time_us for r in all_results])

    print()
    print("RESULTS")
    print("-" * 70)
    print(f"  Total time:       {fmt(total_time)}")
    print(f"  Throughput:       {N / total_time:,.0f} users/sec")
    print(f"  Peak memory:      {peak_mem / 1e6:.0f} MB")
    print()
    print(f"  SARP consistent:  {n_sarp:,} ({n_sarp / N * 100:.1f}%)")
    print(f"  WARP consistent:  {n_warp:,} ({n_warp / N * 100:.1f}%)")
    print(f"  HM efficiency:    mean={hm.mean():.3f}  "
          f"P10={np.percentile(hm, 10):.3f}  P90={np.percentile(hm, 90):.3f}")
    print(f"  Per-user time:    mean={times.mean():.0f}us  "
          f"P50={np.median(times):.0f}us  P99={np.percentile(times, 99):.0f}us")
    print("=" * 70)


if __name__ == "__main__":
    main()
