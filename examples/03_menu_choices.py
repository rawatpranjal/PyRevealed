"""Menu choices: SARP and Houtman-Maks on discrete choice data (no prices)."""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
from pyrevealed import MenuChoiceLog, validate_menu_sarp, compute_menu_efficiency


def simulate_menu_data(n_items=10, n_observations=30, seed=42):
    """Simulate app click-through data: menus of items and user choices."""
    rng = np.random.RandomState(seed)
    menus = []
    choices = []
    for _ in range(n_observations):
        menu_size = rng.randint(2, min(6, n_items + 1))
        menu = frozenset(rng.choice(n_items, menu_size, replace=False).tolist())
        choice = rng.choice(list(menu))
        menus.append(menu)
        choices.append(choice)
    return menus, choices


def main():
    print("Menu Choice Analysis (no prices)")
    print("=" * 50)

    # Simulate 5 users with different choice patterns
    rng = np.random.RandomState(123)
    for user_id in range(5):
        menus, choices = simulate_menu_data(
            n_items=8, n_observations=20, seed=user_id * 100
        )
        log = MenuChoiceLog(menus=menus, choices=choices)

        sarp = validate_menu_sarp(log)
        efficiency = compute_menu_efficiency(log)

        status = "CONSISTENT" if sarp.is_consistent else f"{sarp.num_violations} violations"
        hm = f"{efficiency.fraction:.0%}" if hasattr(efficiency, 'fraction') else "n/a"

        print(f"  User {user_id}: {status:>15}  |  HM efficiency: {hm}")

    print()
    print("SARP checks if item choices form a consistent preference order.")
    print("Houtman-Maks measures what fraction of choices are rationalizable.")


if __name__ == "__main__":
    main()
