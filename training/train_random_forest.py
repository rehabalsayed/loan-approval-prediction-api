import argparse
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from config import (
    RF_PARAMS,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI
)

from utils import (
    split_data,
    calculate_metrics,
    create_confusion_matrix,
    create_roc_curve
)


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    cv,
    n_iter
):

    pipeline = Pipeline([
        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ])

    params = {
        f"model__{k}": v
        for k, v in RF_PARAMS.items()
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        cv=cv,
        n_iter=n_iter,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )

    search.fit(
        X_train,
        y_train
    )

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)

    y_prob = best_model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        y_pred
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="RandomForest"
    ):

        mlflow.set_tag(
            "classifier",
            "RandomForest"
        )

        mlflow.log_params(
            search.best_params_
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.sklearn.log_model(
            best_model,
            "random_forest_model"
        )

        conf_fig = create_confusion_matrix(
            y_test,
            y_pred,
            "Random Forest"
        )

        mlflow.log_figure(
            conf_fig,
            "confusion_matrix.png"
        )

        roc_fig = create_roc_curve(
            y_test,
            y_prob,
            "Random Forest"
        )

        mlflow.log_figure(
            roc_fig,
            "roc_curve.png"
        )

        print("\nBest Parameters:")
        print(search.best_params_)

        print("\nMetrics:")
        print(metrics)


def main(
    cv,
    n_iter
):

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data()

    train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        cv,
        n_iter
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cv",
        type=int,
        default=5
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=20
    )

    args = parser.parse_args()

    main(
        cv=args.cv,
        n_iter=args.n_iter
    )