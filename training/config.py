from pathlib import Path

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    TRAINING_DIR
    / "data"
    / "loan_cleaned.csv"
)


MODELS_DIR = (
    BASE_DIR
    / "models"
)

REPORTS_DIR = (
    BASE_DIR
    / "reports"
)

MLRUNS_DIR = (
    BASE_DIR
    / "mlruns"
)

# ==========================
# MLflow
# ==========================

MLFLOW_TRACKING_URI = (
    f"file:///{MLRUNS_DIR.as_posix()}"
)

EXPERIMENT_NAME = (
    "loan_approval_project"
)

# ==========================
# Train Test Split
# ==========================

TEST_SIZE = 0.20

RANDOM_STATE = 42

# ==========================
# Logistic Regression
# ==========================

LOGISTIC_PARAMS = {

    "C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "solver": [
        "liblinear",
        "lbfgs"
    ]
}

# ==========================
# Random Forest
# ==========================

RF_PARAMS = {

    "n_estimators": [
        100,
        200,
        300,
        500
    ],

    "max_depth": [
        5,
        10,
        15,
        20,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]
}

# ==========================
# XGBoost
# ==========================

XGB_PARAMS = {

    "n_estimators": [
        100,
        200,
        300,
        500
    ],

    "max_depth": [
        3,
        5,
        7,
        10
    ],

    "learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2
    ],

    "subsample": [
        0.8,
        0.9,
        1.0
    ]
}