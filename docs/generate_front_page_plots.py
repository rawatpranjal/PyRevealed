#!/usr/bin/env python3
"""Generate static plot images for the front page documentation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we're using a non-interactive backend
plt.switch_backend('Agg')

from pyrevealed import BehaviorLog
from pyrevealed.viz import (
    plot_budget_sets,
    plot_aei_distribution,
    plot_power_analysis,
    plot_ccei_sensitivity,
)


def main():
    # Output directory
    output_dir = Path(__file__).parent / "_static"
    output_dir.mkdir(exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Budget Sets: 2 goods, 4 observations for clear visualization
    print("Generating budget sets plot...")
    log_2d = BehaviorLog(
        cost_vectors=np.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 1.5],
            [1.2, 1.8],
        ]),
        action_vectors=np.array([
            [4, 3],
            [3, 4],
            [3, 3],
            [4, 2],
        ])
    )
    fig, ax = plot_budget_sets(log_2d, figsize=(6, 6))
    fig.savefig(output_dir / "front_budget_sets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. AEI Distribution: sample population scores
    print("Generating AEI distribution plot...")
    # Generate realistic distribution: most users are fairly consistent
    aei_scores = np.concatenate([
        np.random.beta(8, 2, size=60),  # High consistency cluster
        np.random.beta(3, 2, size=30),  # Moderate consistency
        np.random.beta(2, 5, size=10),  # Low consistency outliers
    ])
    aei_scores = np.clip(aei_scores, 0.3, 1.0)  # Clip to realistic range
    fig, ax = plot_aei_distribution(aei_scores.tolist(), bins=25, figsize=(8, 5))
    fig.savefig(output_dir / "front_aei_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Power Analysis: use a moderate-sized dataset
    print("Generating power analysis plot...")
    log_power = BehaviorLog(
        cost_vectors=np.random.rand(15, 4) + 0.5,
        action_vectors=np.random.rand(15, 4) * 5 + 0.5,
    )
    fig, ax = plot_power_analysis(log_power, n_simulations=200, figsize=(8, 5))
    fig.savefig(output_dir / "front_power_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. CCEI Sensitivity: show how removing outliers improves consistency
    print("Generating CCEI sensitivity plot...")
    # Create data with some inconsistencies
    log_sensitivity = BehaviorLog(
        cost_vectors=np.array([
            [1.0, 2.0, 1.5],
            [2.0, 1.0, 1.5],
            [1.5, 1.5, 1.0],
            [1.2, 1.8, 1.3],
            [1.8, 1.2, 1.3],
            [1.0, 1.0, 2.0],
            [1.3, 1.7, 1.2],
            [1.7, 1.3, 1.2],
            [1.4, 1.4, 1.4],
            [1.1, 1.9, 1.1],
        ]),
        action_vectors=np.array([
            [2, 4, 1],
            [4, 2, 1],
            [3, 2, 2],
            [3, 3, 1],
            [3, 3, 1],
            [2, 2, 3],
            [2, 4, 1],
            [4, 2, 1],
            [3, 3, 1],
            [2, 4, 1],
        ])
    )
    fig, ax = plot_ccei_sensitivity(log_sensitivity, max_remove=4, figsize=(8, 5))
    fig.savefig(output_dir / "front_ccei_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All plots saved to {output_dir}")

    # List generated files
    for f in sorted(output_dir.glob("front_*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
