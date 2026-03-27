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

# LightGBM hyperparameters (shared across all benchmarks)
LGBM_CLASSIFIER_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 100,
    "random_state": SEED,
}

LGBM_REGRESSOR_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 100,
    "random_state": SEED,
}

# Engine metrics to compute
BUDGET_ENGINE_METRICS = ["garp", "ccei", "mpi", "harp", "hm", "vei"]
