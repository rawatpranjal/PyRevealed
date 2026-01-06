"""Simulation data generators for testing and demos.

Provides synthetic data generators for:
- Risk preference scenarios (gamblers vs investors)
- Spatial preference scenarios (ideal point users)
- Separability scenarios (superapp cannibalization)
- General rational/irrational consumer data
"""

from sim.risk_scenarios import (
    generate_gambler,
    generate_investor,
    generate_risk_neutral,
    generate_mixed_risk_population,
    generate_lottery_choice_experiment,
)
from sim.spatial_scenarios import (
    generate_euclidean_user,
    generate_noisy_user,
    generate_multi_ideal_user,
    generate_recommendation_scenario,
    generate_dating_app_scenario,
)
from sim.separability_scenarios import (
    generate_separable_superapp,
    generate_cannibalized_superapp,
    generate_mixed_superapp,
    generate_amazon_scenario,
)
from sim.generators import (
    generate_rational_data,
    generate_irrational_data,
    generate_garp_violation_cycle,
    generate_efficiency_data,
)

__all__ = [
    # Risk scenarios
    "generate_gambler",
    "generate_investor",
    "generate_risk_neutral",
    "generate_mixed_risk_population",
    "generate_lottery_choice_experiment",
    # Spatial scenarios
    "generate_euclidean_user",
    "generate_noisy_user",
    "generate_multi_ideal_user",
    "generate_recommendation_scenario",
    "generate_dating_app_scenario",
    # Separability scenarios
    "generate_separable_superapp",
    "generate_cannibalized_superapp",
    "generate_mixed_superapp",
    "generate_amazon_scenario",
    # General generators
    "generate_rational_data",
    "generate_irrational_data",
    "generate_garp_violation_cycle",
    "generate_efficiency_data",
]
