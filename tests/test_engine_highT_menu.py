"""High-T stress tests for Engine (menu/discrete choice).

Cases:
- Many menus (T≈3000) with consistent choices from a fixed utility vector
- Same with an isolated WARP reversal injected => small HM removal

Marked slow to avoid default runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from prefgraph.engine import Engine


def _consistent_menu_user(T: int, n_items: int, seed: int = 123):
    """Generate SARP/WARP-consistent menus by always picking argmax utility.

    - Random utility vector u for items
    - Each menu is a random subset of items of size in [2, min(10, n_items)]
    - Choice is argmax u over the menu
    """
    rng = np.random.RandomState(seed)
    u = rng.rand(n_items).astype(np.float64)

    menus = []
    choices = []
    for _ in range(T):
        msize = int(rng.randint(2, min(10, n_items) + 1))
        menu = rng.choice(n_items, size=msize, replace=False).tolist()
        # Choose the item with max utility in this menu
        best = max(menu, key=lambda i: u[i])
        menus.append(menu)
        choices.append(best)
    return menus, choices


@pytest.mark.slow
def test_highT_consistent_menu_T3000():
    T, n_items = 3000, 50
    menus, choices = _consistent_menu_user(T, n_items, seed=2024)

    e = Engine()
    [r] = e.analyze_menus([(menus, choices, n_items)])

    assert r.is_sarp is True
    assert r.is_warp is True
    # For menus, HM counts items, not menu observations
    assert r.hm_total == n_items and r.hm_consistent == n_items


@pytest.mark.slow
def test_highT_menu_with_isolated_warp_reversal():
    """Inject two-menu direct reversal into an otherwise consistent user.

    Expect:
    - WARP and SARP violations
    - HM consistent = T-1 (remove either of the two reversal observations)
    """
    T, n_items = 5000, 200
    # Build consistent menus that avoid items {0,1} entirely
    rng = np.random.RandomState(777)
    menus = []
    choices = []
    u = rng.rand(n_items).astype(np.float64)
    for _ in range(T - 2):
        # Draw from items 2..n_items-1 only
        pool = list(range(2, n_items))
        msize = int(rng.randint(2, min(10, n_items - 2) + 1))
        menu = rng.choice(pool, size=msize, replace=False).tolist()
        best = max(menu, key=lambda i: u[i])
        menus.append(menu)
        choices.append(best)

    # Inject isolated direct reversal on items {0,1} (not used elsewhere)
    menus = [[0, 1], [1, 0]] + menus
    choices = [0, 1] + choices

    e = Engine()
    [r] = e.analyze_menus([(menus, choices, n_items)])

    assert r.is_warp is False
    assert r.is_sarp is False
    # Because items {0,1} are isolated from the rest, removing exactly
    # one of them suffices; HM should be n_items-1 exactly under item-count HM.
    assert r.hm_total == n_items and r.hm_consistent == n_items - 1
