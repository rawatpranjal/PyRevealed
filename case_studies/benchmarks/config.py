"""Benchmark configuration constants."""

SEED = 42
TRAIN_FRACTION = 0.7  # First 70% of observations for features, last 30% for targets
MIN_OBS_BUDGET = 10
MIN_OBS_MENU = 5
MIN_TRAIN_BUDGET = 5
MIN_TEST_BUDGET = 3
MIN_TRAIN_MENU = 3
MIN_TEST_MENU = 2

# Engine metrics to compute
BUDGET_ENGINE_METRICS = ["garp", "ccei", "mpi", "harp", "hm", "vei"]
