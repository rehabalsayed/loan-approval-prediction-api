import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from config import (
    DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE
)


def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    return df


def prepare_data():

    df = load_data()

    X = df.drop(
        columns=["loan_status"]
    )

    y = df["loan_status"]

    return X, y


def split_data():

    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        stratify=y,

        random_state=RANDOM_STATE
    )

    return (

        X_train,
        X_test,

        y_train,
        y_test
    )


def calculate_metrics(
    y_true,
    y_pred
):

    metrics = {

        "accuracy":
        accuracy_score(
            y_true,
            y_pred
        ),

        "precision":
        precision_score(
            y_true,
            y_pred
        ),

        "recall":
        recall_score(
            y_true,
            y_pred
        ),

        "f1_score":
        f1_score(
            y_true,
            y_pred
        )
    }

    return metrics


def create_confusion_matrix(
    y_true,
    y_pred,
    model_name
):

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(

        confusion_matrix(
            y_true,
            y_pred
        ),

        annot=True,

        fmt='d',

        cmap='Blues'
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    return plt.gcf()


def create_roc_curve(
    y_true,
    y_prob,
    model_name
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"{model_name} ROC Curve"
    )

    plt.legend()

    return plt.gcf()


