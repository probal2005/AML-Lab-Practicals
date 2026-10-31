"""
PRACTICAL 10: FEDERATED LEARNING

Simulated federated learning using the Breast Cancer Wisconsin dataset.

Four clients train independent Logistic Regression models.
Their predictions are combined using majority voting.

NOTE:
This is a simulation of federated learning through prediction aggregation.
It does not implement parameter aggregation such as FedAvg.
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
NUM_CLIENTS = 4


# ============================================================
# 2. OUTPUT LOGGING
# ============================================================

class OutputLogger:
    """Display program output and save it to a text file."""

    def __init__(self, output_file):
        self.output_file = output_file
        self.file = open(
            output_file,
            "w",
            encoding="utf-8"
        )

    def write(self, message=""):
        print(message)
        self.file.write(str(message) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


logger = OutputLogger(OUTPUT_DIR / "experiment_output.txt")


def save_dataframe(dataframe, filename):
    """Save a DataFrame as a CSV file."""
    dataframe.to_csv(
        OUTPUT_DIR / filename,
        index=False
    )


# ============================================================
# 3. MAIN EXPERIMENT
# ============================================================

def run_experiment():

    logger.write("=" * 68)
    logger.write("PRACTICAL 10: FEDERATED LEARNING")
    logger.write("Breast Cancer Classification Using Multiple Clients")
    logger.write("=" * 68)

    # --------------------------------------------------------
    # STEP 1: LOAD DATASET
    # --------------------------------------------------------

    logger.write("\n1. LOADING BREAST CANCER WISCONSIN DATASET")

    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(
        dataset.target,
        name="target"
    )

    logger.write(f"Total samples       : {X.shape[0]}")
    logger.write(f"Total features      : {X.shape[1]}")
    logger.write(f"Number of classes   : {len(dataset.target_names)}")
    logger.write(f"Class names         : {list(dataset.target_names)}")

    class_distribution = (
        y.value_counts()
        .sort_index()
        .rename(index={
            0: dataset.target_names[0],
            1: dataset.target_names[1]
        })
    )

    logger.write("\nDataset class distribution:")
    logger.write(class_distribution.to_string())

    dataset_frame = X.copy()
    dataset_frame["target"] = y
    dataset_frame["target_name"] = y.map(
        lambda value: dataset.target_names[value]
    )

    save_dataframe(
        dataset_frame,
        "breast_cancer_dataset.csv"
    )

    # --------------------------------------------------------
    # STEP 2: SPLIT TRAINING AND TESTING DATA
    # --------------------------------------------------------

    logger.write("\n2. SPLITTING TRAINING AND TESTING DATA")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.write(f"Training samples    : {len(X_train)}")
    logger.write(f"Testing samples     : {len(X_test)}")

    # --------------------------------------------------------
    # STEP 3: FEATURE SCALING
    # --------------------------------------------------------

    logger.write("\n3. PERFORMING FEATURE SCALING")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.write("Feature scaling completed.")

    # --------------------------------------------------------
    # STEP 4: DIVIDE DATA INTO CLIENTS
    # --------------------------------------------------------

    logger.write("\n4. DIVIDING TRAINING DATA INTO CLIENTS")

    client_X = np.array_split(
        X_train_scaled,
        NUM_CLIENTS
    )

    client_y = np.array_split(
        y_train.to_numpy(),
        NUM_CLIENTS
    )

    client_models = []
    client_predictions = []
    client_accuracies = []

    for client_index in range(NUM_CLIENTS):

        client_number = client_index + 1

        logger.write(f"\nClient {client_number}")
        logger.write("-" * 35)

        X_local = client_X[client_index]
        y_local = client_y[client_index]

        logger.write(f"Local training samples: {len(X_local)}")
        logger.write(
            "Local class distribution: "
            + str(dict(zip(
                *np.unique(y_local, return_counts=True)
            )))
        )

        if len(np.unique(y_local)) < 2:
            raise ValueError(
                f"Client {client_number} contains only one class. "
                "Logistic Regression requires at least two classes."
            )

        # ----------------------------------------------------
        # STEP 5: TRAIN LOCAL MODEL
        # ----------------------------------------------------

        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )

        model.fit(X_local, y_local)

        client_models.append(model)

        logger.write("Local Logistic Regression model trained.")

        # ----------------------------------------------------
        # STEP 6: EVALUATE CLIENT MODEL
        # ----------------------------------------------------

        predictions = model.predict(X_test_scaled)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        client_predictions.append(predictions)
        client_accuracies.append(accuracy)

        logger.write(
            f"Client {client_number} Accuracy: {accuracy:.4f}"
        )

        logger.write(
            classification_report(
                y_test,
                predictions,
                target_names=dataset.target_names,
                zero_division=0
            )
        )

    # --------------------------------------------------------
    # STEP 7: COLLECT CLIENT PREDICTIONS
    # --------------------------------------------------------

    logger.write("\n5. COLLECTING CLIENT PREDICTIONS")

    all_predictions = np.asarray(
        client_predictions
    )

    logger.write(
        f"Prediction matrix shape: {all_predictions.shape}"
    )

    # --------------------------------------------------------
    # STEP 8: MAJORITY VOTING
    # --------------------------------------------------------

    logger.write("\n6. COMBINING PREDICTIONS USING MAJORITY VOTING")

    global_predictions = []

    for sample_index in range(len(y_test)):

        votes = all_predictions[:, sample_index]

        vote_counts = np.bincount(
            votes.astype(int),
            minlength=len(dataset.target_names)
        )

        # np.argmax chooses the first class in case of a tie.
        predicted_class = int(np.argmax(vote_counts))

        global_predictions.append(predicted_class)

    global_predictions = np.asarray(
        global_predictions
    )

    global_accuracy = accuracy_score(
        y_test,
        global_predictions
    )

    logger.write(
        f"Global Model Accuracy: {global_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # STEP 9: GLOBAL CLASSIFICATION REPORT
    # --------------------------------------------------------

    logger.write("\n7. GLOBAL CLASSIFICATION REPORT")

    global_report = classification_report(
        y_test,
        global_predictions,
        target_names=dataset.target_names,
        zero_division=0
    )

    logger.write(global_report)

    with open(
        OUTPUT_DIR / "global_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as report_file:
        report_file.write(global_report)

    # --------------------------------------------------------
    # STEP 10: CONFUSION MATRIX
    # --------------------------------------------------------

    logger.write("\n8. GLOBAL CONFUSION MATRIX")

    global_cm = confusion_matrix(
        y_test,
        global_predictions,
        labels=np.arange(len(dataset.target_names))
    )

    logger.write(str(global_cm))

    cm_frame = pd.DataFrame(
        global_cm,
        index=[
            f"Actual {name}"
            for name in dataset.target_names
        ],
        columns=[
            f"Predicted {name}"
            for name in dataset.target_names
        ]
    )

    cm_frame.to_csv(
        OUTPUT_DIR / "global_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=global_cm,
        display_labels=dataset.target_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "Federated Learning - Global Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "global_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # STEP 11: CLIENT AND GLOBAL ACCURACY COMPARISON
    # --------------------------------------------------------

    logger.write("\n9. CLIENT VS GLOBAL ACCURACY")

    comparison_names = [
        f"Client {i + 1}"
        for i in range(NUM_CLIENTS)
    ]

    comparison_accuracies = list(client_accuracies)

    comparison_names.append("Global Model")
    comparison_accuracies.append(global_accuracy)

    comparison_df = pd.DataFrame({
        "Model": comparison_names,
        "Accuracy": comparison_accuracies,
        "Accuracy_Percentage": [
            accuracy * 100
            for accuracy in comparison_accuracies
        ]
    })

    logger.write(
        comparison_df.to_string(index=False)
    )

    save_dataframe(
        comparison_df,
        "client_accuracy_results.csv"
    )

    # --------------------------------------------------------
    # STEP 12: CLIENT ACCURACY BAR CHART
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        comparison_names,
        comparison_accuracies
    )

    ax.set_title(
        "Client and Global Model Accuracy Comparison"
    )

    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    for bar, accuracy in zip(
        bars,
        comparison_accuracies
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom"
        )

    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "federated_learning_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # STEP 13: SAVE CLIENT PREDICTIONS
    # --------------------------------------------------------

    prediction_data = pd.DataFrame({
        "Actual_Class": y_test.to_numpy(),
        "Actual_Class_Name": [
            dataset.target_names[value]
            for value in y_test.to_numpy()
        ],
        "Global_Prediction": global_predictions,
        "Global_Prediction_Name": [
            dataset.target_names[value]
            for value in global_predictions
        ]
    })

    for client_index in range(NUM_CLIENTS):
        prediction_data[
            f"Client_{client_index + 1}_Prediction"
        ] = client_predictions[client_index]

    save_dataframe(
        prediction_data,
        "client_predictions.csv"
    )

    # --------------------------------------------------------
    # STEP 14: SAVE CLIENT CLASSIFICATION REPORTS
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "client_classification_reports.txt",
        "w",
        encoding="utf-8"
    ) as report_file:

        for client_index in range(NUM_CLIENTS):

            report_file.write(
                f"\nCLIENT {client_index + 1} CLASSIFICATION REPORT\n"
            )

            report_file.write("=" * 55 + "\n")

            report_file.write(
                classification_report(
                    y_test,
                    client_predictions[client_index],
                    target_names=dataset.target_names,
                    zero_division=0
                )
            )

            report_file.write("\n")

    # --------------------------------------------------------
    # STEP 15: FINAL SUMMARY
    # --------------------------------------------------------

    logger.write("\n10. FINAL EXPERIMENT SUMMARY")
    logger.write("=" * 55)

    logger.write(f"Dataset: {dataset.DESCR.splitlines()[0]}")
    logger.write(f"Total clients: {NUM_CLIENTS}")
    logger.write(f"Training samples: {len(X_train)}")
    logger.write(f"Testing samples: {len(X_test)}")

    for client_index, accuracy in enumerate(
        client_accuracies
    ):
        logger.write(
            f"Client {client_index + 1} Accuracy: "
            f"{accuracy:.4f}"
        )

    logger.write(
        f"Global Majority Voting Accuracy: {global_accuracy:.4f}"
    )

    logger.write(
        "\nFederated learning simulation completed successfully."
    )

    logger.write(
        "Raw client training partitions were not exported "
        "or combined into a central training dataset."
    )

    logger.write(
        "\nIMPORTANT: This experiment demonstrates local model "
        "training and prediction aggregation. It is not a full "
        "federated optimization protocol such as FedAvg."
    )

    logger.write("\nGenerated output files:")

    for file_path in sorted(OUTPUT_DIR.iterdir()):
        if file_path.is_file():
            logger.write(
                f"- {file_path.name} ({file_path.stat().st_size} bytes)"
            )


# ============================================================
# 4. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        run_experiment()

    except Exception as error:

        error_message = (
            f"\nEXPERIMENT FAILED: {error}\n\n"
            f"{traceback.format_exc()}"
        )

        print(error_message)

        with open(
            OUTPUT_DIR / "error_log.txt",
            "w",
            encoding="utf-8"
        ) as error_file:
            error_file.write(error_message)

        sys.exit(1)

    finally:
        logger.close()
