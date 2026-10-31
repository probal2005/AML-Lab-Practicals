"""
Practical 9: Federated Learning System using a Machine Learning Model
Course Outcome: CO4

Five simulated clients train independent Logistic Regression models.
Their predictions are combined using majority voting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import mode

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
NUM_CLIENTS = 5
MAX_ITERATIONS = 1000

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 65)
print("PRACTICAL 9: FEDERATED LEARNING")
print("=" * 65)

data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print("\nDATASET INFORMATION")
print("-" * 65)
print("Dataset: Breast Cancer Wisconsin Dataset")
print("Dataset Shape:", X.shape)
print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])
print("Target Classes:", target_names)


# ------------------------------------------------------------
# 3. CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("\nFIRST FIVE DATASET RECORDS")
print("-" * 65)
print(df.head().to_string(index=False))

df.head().to_csv(OUTPUT_DIR / "dataset_preview.csv", index=False)


# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nDATA SPLITTING")
print("-" * 65)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ------------------------------------------------------------
# 5. FEATURE STANDARDIZATION
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFEATURE STANDARDIZATION")
print("-" * 65)
print("Training and testing features standardized successfully.")


# ------------------------------------------------------------
# 6. DIVIDE TRAINING DATA INTO CLIENTS
# ------------------------------------------------------------

X_clients = np.array_split(X_train_scaled, NUM_CLIENTS)
y_clients = np.array_split(y_train, NUM_CLIENTS)

print("\nCLIENT DATA DISTRIBUTION")
print("-" * 65)

for i in range(NUM_CLIENTS):
    print(
        f"Client {i + 1}: "
        f"{len(X_clients[i])} training samples"
    )


# ------------------------------------------------------------
# 7. TRAIN LOCAL LOGISTIC REGRESSION MODELS
# ------------------------------------------------------------

client_models = []
client_results = []
client_predictions = []

print("\nLOCAL CLIENT MODEL TRAINING")
print("-" * 65)

for i in range(NUM_CLIENTS):

    local_model = LogisticRegression(
        max_iter=MAX_ITERATIONS,
        random_state=RANDOM_STATE,
    )

    local_model.fit(X_clients[i], y_clients[i])

    client_models.append(local_model)

    predictions = local_model.predict(X_test_scaled)
    client_predictions.append(predictions)

    accuracy = accuracy_score(y_test, predictions)

    client_results.append(
        {
            "Client": f"Client {i + 1}",
            "Training Samples": len(X_clients[i]),
            "Accuracy": accuracy,
        }
    )

    print(
        f"Client {i + 1} Accuracy: "
        f"{accuracy:.4f}"
    )


# ------------------------------------------------------------
# 8. COLLECT CLIENT PREDICTIONS
# ------------------------------------------------------------

all_client_predictions = np.array(client_predictions)

print("\nCLIENT PREDICTION COLLECTION")
print("-" * 65)
print("Prediction Matrix Shape:", all_client_predictions.shape)
print("Number of Client Models:", len(client_models))


# ------------------------------------------------------------
# 9. MAJORITY VOTING AGGREGATION
# ------------------------------------------------------------

global_predictions = mode(
    all_client_predictions,
    axis=0,
    keepdims=False,
).mode

global_predictions = np.asarray(global_predictions, dtype=int)

print("\nFEDERATED MAJORITY VOTING")
print("-" * 65)
print("Predictions combined successfully.")
print("Total Global Predictions:", len(global_predictions))


# ------------------------------------------------------------
# 10. EVALUATE GLOBAL FEDERATED SYSTEM
# ------------------------------------------------------------

global_accuracy = accuracy_score(
    y_test,
    global_predictions,
)

global_confusion_matrix = confusion_matrix(
    y_test,
    global_predictions,
    labels=[0, 1],
)

report_dict = classification_report(
    y_test,
    global_predictions,
    labels=[0, 1],
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)

print("\nFEDERATED GLOBAL MODEL EVALUATION")
print("-" * 65)
print(f"Global Model Accuracy: {global_accuracy:.4f}")

print("\nCONFUSION MATRIX")
print(global_confusion_matrix)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        global_predictions,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0,
    )
)


# ------------------------------------------------------------
# 11. CREATE AND SAVE RESULT TABLES
# ------------------------------------------------------------

results_df = pd.DataFrame(client_results)

global_row = pd.DataFrame(
    [
        {
            "Client": "Federated Global Model",
            "Training Samples": len(X_train_scaled),
            "Accuracy": global_accuracy,
        }
    ]
)

final_results = pd.concat(
    [results_df, global_row],
    ignore_index=True,
)

final_results.to_csv(
    OUTPUT_DIR / "federated_accuracy_comparison.csv",
    index=False,
)

pd.DataFrame(
    global_confusion_matrix,
    index=["Actual Malignant", "Actual Benign"],
    columns=["Predicted Malignant", "Predicted Benign"],
).to_csv(OUTPUT_DIR / "federated_confusion_matrix.csv")

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "federated_classification_report.csv"
)

prediction_results = pd.DataFrame(
    {
        "Actual Class": y_test,
        "Federated Predicted Class": global_predictions,
        "Correct Prediction": y_test == global_predictions,
    }
)

prediction_results.to_csv(
    OUTPUT_DIR / "federated_predictions.csv",
    index=False,
)

print("\nFINAL ACCURACY COMPARISON")
print("-" * 65)
print(final_results.to_string(index=False))


# ------------------------------------------------------------
# 12. VISUALIZATION 1: ACCURACY COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    final_results["Client"],
    final_results["Accuracy"],
)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Client Models vs Federated Global Model Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "federated_accuracy_comparison.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# 13. VISUALIZATION 2: CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.imshow(global_confusion_matrix, interpolation="nearest")
plt.title("Federated Global Model Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(target_names))

plt.xticks(tick_marks, target_names, rotation=20)
plt.yticks(tick_marks, target_names)

threshold = global_confusion_matrix.max() / 2

for i in range(global_confusion_matrix.shape[0]):
    for j in range(global_confusion_matrix.shape[1]):
        plt.text(
            j,
            i,
            str(global_confusion_matrix[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if global_confusion_matrix[i, j] > threshold
                else "black"
            ),
            fontsize=12,
        )

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "federated_confusion_matrix.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# 14. VISUALIZATION 3: CLIENT TRAINING DISTRIBUTION
# ------------------------------------------------------------

client_names = [
    f"Client {i + 1}"
    for i in range(NUM_CLIENTS)
]

client_sample_counts = [
    len(client_data)
    for client_data in X_clients
]

plt.figure(figsize=(8, 5))

plt.bar(client_names, client_sample_counts)

plt.xlabel("Federated Clients")
plt.ylabel("Training Samples")
plt.title("Training Data Distribution Across Clients")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "client_training_distribution.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# 15. SAVE EXECUTION SUMMARY
# ------------------------------------------------------------

summary_lines = [
    "PRACTICAL 9: FEDERATED LEARNING",
    "=" * 50,
    f"Dataset: Breast Cancer Wisconsin Dataset",
    f"Total Samples: {len(X)}",
    f"Total Features: {X.shape[1]}",
    f"Training Samples: {len(X_train)}",
    f"Testing Samples: {len(X_test)}",
    f"Number of Clients: {NUM_CLIENTS}",
    f"Model: Logistic Regression",
    "Aggregation: Majority Voting",
    f"Global Accuracy: {global_accuracy:.4f}",
    "",
    "CLIENT ACCURACY RESULTS",
    final_results.to_string(index=False),
    "",
    "GLOBAL CONFUSION MATRIX",
    str(global_confusion_matrix),
    "",
    "CLASSIFICATION REPORT",
    classification_report(
        y_test,
        global_predictions,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0,
    ),
    "",
    "NOTE:",
    "This experiment simulates federated learning using local",
    "client models and prediction-level majority voting.",
]

(OUTPUT_DIR / "experiment_output.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\nOUTPUT FILES")
print("-" * 65)

for output_file in sorted(OUTPUT_DIR.iterdir()):
    if output_file.is_file():
        print(output_file.name)

print("\n" + "=" * 65)
print("FEDERATED LEARNING EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)
