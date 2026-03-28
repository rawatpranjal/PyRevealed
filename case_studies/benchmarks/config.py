"""Benchmark configuration constants."""

SEED = 42
N_FOLDS = 5
TRAIN_FRACTION = 0.7  # First 70% of observations for features, last 30% for targets
MIN_OBS_BUDGET = 10  # Minimum observations per user for budget datasets
MIN_OBS_MENU = 5  # Minimum sessions per user for menu datasets
MIN_TRAIN_BUDGET = 5  # Minimum training observations for budget datasets
MIN_TEST_BUDGET = 3  # Minimum test observations for budget datasets
MIN_TRAIN_MENU = 3  # Minimum training observations for menu datasets
MIN_TEST_MENU = 2  # Minimum test observations for menu datasets

# CatBoost hyperparameters (shared across all benchmarks)
CATBOOST_CLASSIFIER_PARAMS = {
    "depth": 5,
    "learning_rate": 0.03,
    "iterations": 200,
    "l2_leaf_reg": 5.0,
    "rsm": 0.6,
    "subsample": 0.7,
    "random_seed": SEED,
    "verbose": 0,
    "eval_metric": "AUC",
}

CATBOOST_REGRESSOR_PARAMS = {
    "depth": 5,
    "learning_rate": 0.03,
    "iterations": 200,
    "l2_leaf_reg": 5.0,
    "rsm": 0.6,
    "subsample": 0.7,
    "random_seed": SEED,
    "verbose": 0,
    "eval_metric": "RMSE",
}

# Engine metrics to compute
BUDGET_ENGINE_METRICS = ["garp", "ccei", "mpi", "harp", "hm", "vei"]
