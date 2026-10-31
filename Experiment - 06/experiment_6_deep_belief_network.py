# ============================================================
# PRACTICAL 6: DEEP BELIEF NETWORK USING STACKED RBM LAYERS
# Mapped Course Outcome: CO4
# Dataset: Scikit-learn Digits Dataset
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print_section("1. LOADING DIGITS DATASET")

digits = load_digits()

X = digits.data
y = digits.target

print("Dataset shape:", X.shape)
print("Target classes:", np.unique(y))
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Number of classes:", len(np.unique(y)))

dataset_df = pd.DataFrame(
    X,
    columns=[f"pixel_{i}" for i in range(X.shape[1])]
)

dataset_df["target"] = y
dataset_df.to_csv(
    os.path.join(OUTPUT_DIR, "digits_dataset.csv"),
    index=False
)

# ------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------------

print_section("2. TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------
# 4. DEFINE DBN-LIKE MODEL USING TWO RBM LAYERS
# ------------------------------------------------------------

print_section("3. DEFINING STACKED RBM MODEL")

# BernoulliRBM is used as a feature-learning layer.
# The first RBM learns 64 hidden features.
# The second RBM learns 32 higher-level features.

rbm1 = BernoulliRBM(
    n_components=64,
    learning_rate=0.05,
    n_iter=30,
    batch_size=10,
    random_state=RANDOM_STATE,
    verbose=False
)

rbm2 = BernoulliRBM(
    n_components=32,
    learning_rate=0.05,
    n_iter=30,
    batch_size=10,
    random_state=RANDOM_STATE,
    verbose=False
)

# Scaling is placed inside the pipeline to prevent data leakage
# during cross-validation.

dbn_model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("rbm1", rbm1),
    ("rbm2", rbm2),
    ("classifier", LogisticRegression(max_iter=5000))
])

# ------------------------------------------------------------
# 5. DEFINE RAW DATA BASELINE MODEL
# ------------------------------------------------------------

print_section("4. DEFINING BASELINE MODEL")

raw_model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

print("DBN architecture:")
print("Input features: 64")
print("First RBM hidden components: 64")
print("Second RBM hidden components: 32")
print("Output classes: 10")
print("Final classifier: Logistic Regression")

# ------------------------------------------------------------
# 6. STRATIFIED CROSS-VALIDATION
# ------------------------------------------------------------

print_section("5. STRATIFIED CROSS-VALIDATION")

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted"
}

models = {
    "Logistic Regression on Raw Data": raw_model,
    "DBN Features + Logistic Regression": dbn_model
}

cv_records = []
cv_summary_records = []

for model_name, model in models.items():

    print(f"\nEvaluating: {model_name}")

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    for fold_index in range(CV_FOLDS):
        cv_records.append({
            "Model": model_name,
            "Fold": fold_index + 1,
            "Accuracy": cv_results["test_accuracy"][fold_index],
            "Precision": cv_results["test_precision_weighted"][fold_index],
            "Recall": cv_results["test_recall_weighted"][fold_index],
            "F1_Score": cv_results["test_f1_weighted"][fold_index]
        })

    for metric in scoring:
        values = cv_results[f"test_{metric}"]

        cv_summary_records.append({
            "Model": model_name,
            "Metric": metric,
            "Mean": np.mean(values),
            "Standard Deviation": np.std(values),
            "Minimum": np.min(values),
            "Maximum": np.max(values)
        })

        print(
            f"{metric}: "
            f"Mean={np.mean(values):.4f}, "
            f"Std={np.std(values):.4f}"
        )

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(
    os.path.join(OUTPUT_DIR, "cross_validation_results.csv"),
    index=False
)

cv_summary_df = pd.DataFrame(cv_summary_records)
cv_summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "cross_validation_summary.csv"),
    index=False
)

# ------------------------------------------------------------
# 7. TRAIN DBN MODEL
# ------------------------------------------------------------

print_section("6. TRAINING FIRST AND SECOND RBM LAYERS")

print("Training the complete DBN pipeline...")

dbn_model.fit(X_train, y_train)

print("First RBM layer trained successfully.")
print("Second RBM layer trained successfully.")
print("Logistic Regression classifier trained successfully.")

# ------------------------------------------------------------
# 8. EXTRACT LEARNED FEATURES
# ------------------------------------------------------------

print_section("7. EXTRACTING DBN LEARNED FEATURES")

X_train_scaled = dbn_model.named_steps["scaler"].transform(X_train)
X_test_scaled = dbn_model.named_steps["scaler"].transform(X_test)

X_train_rbm1 = dbn_model.named_steps["rbm1"].transform(
    X_train_scaled
)

X_test_rbm1 = dbn_model.named_steps["rbm1"].transform(
    X_test_scaled
)

X_train_features = dbn_model.named_steps["rbm2"].transform(
    X_train_rbm1
)

X_test_features = dbn_model.named_steps["rbm2"].transform(
    X_test_rbm1
)

print("Original training features:", X_train.shape)
print("First RBM training features:", X_train_rbm1.shape)
print("Second RBM training features:", X_train_features.shape)

print("Original testing features:", X_test.shape)
print("DBN testing features:", X_test_features.shape)

# Save extracted feature representations.

pd.DataFrame(X_train_features).to_csv(
    os.path.join(OUTPUT_DIR, "dbn_training_features.csv"),
    index=False
)

pd.DataFrame(X_test_features).to_csv(
    os.path.join(OUTPUT_DIR, "dbn_testing_features.csv"),
    index=False
)

# ------------------------------------------------------------
# 9. TRAIN BASELINE MODEL
# ------------------------------------------------------------

print_section("8. TRAINING BASELINE LOGISTIC REGRESSION")

raw_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 10. PREDICTIONS
# ------------------------------------------------------------

print_section("9. MAKING PREDICTIONS")

dbn_predictions = dbn_model.predict(X_test)
raw_predictions = raw_model.predict(X_test)

print("DBN predictions generated.")
print("Baseline predictions generated.")

# ------------------------------------------------------------
# 11. PERFORMANCE EVALUATION
# ------------------------------------------------------------

print_section("10. PERFORMANCE EVALUATION")

def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


dbn_metrics = calculate_metrics(y_test, dbn_predictions)
raw_metrics = calculate_metrics(y_test, raw_predictions)

results_df = pd.DataFrame([
    {
        "Model": "Logistic Regression on Raw Data",
        **raw_metrics
    },
    {
        "Model": "DBN Features + Logistic Regression",
        **dbn_metrics
    }
])

print("\nTest Set Performance:")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 12. CLASSIFICATION REPORTS
# ------------------------------------------------------------

print_section("11. CLASSIFICATION REPORTS")

target_names = [str(i) for i in digits.target_names]

dbn_report = classification_report(
    y_test,
    dbn_predictions,
    labels=list(range(10)),
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

raw_report = classification_report(
    y_test,
    raw_predictions,
    labels=list(range(10)),
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

print("\nDBN Classification Report:")
print(
    classification_report(
        y_test,
        dbn_predictions,
        labels=list(range(10)),
        target_names=target_names,
        zero_division=0
    )
)

print("\nRaw Logistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        raw_predictions,
        labels=list(range(10)),
        target_names=target_names,
        zero_division=0
    )
)

pd.DataFrame(dbn_report).transpose().to_csv(
    os.path.join(OUTPUT_DIR, "dbn_classification_report.csv")
)

pd.DataFrame(raw_report).transpose().to_csv(
    os.path.join(OUTPUT_DIR, "raw_classification_report.csv")
)

# ------------------------------------------------------------
# 13. CONFUSION MATRICES
# ------------------------------------------------------------

print_section("12. CONFUSION MATRICES")

dbn_cm = confusion_matrix(
    y_test,
    dbn_predictions,
    labels=list(range(10))
)

raw_cm = confusion_matrix(
    y_test,
    raw_predictions,
    labels=list(range(10))
)

pd.DataFrame(
    dbn_cm,
    index=[f"Actual_{i}" for i in range(10)],
    columns=[f"Predicted_{i}" for i in range(10)]
).to_csv(
    os.path.join(OUTPUT_DIR, "dbn_confusion_matrix.csv")
)

pd.DataFrame(
    raw_cm,
    index=[f"Actual_{i}" for i in range(10)],
    columns=[f"Predicted_{i}" for i in range(10)]
).to_csv(
    os.path.join(OUTPUT_DIR, "raw_confusion_matrix.csv")
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ConfusionMatrixDisplay(
    confusion_matrix=dbn_cm,
    display_labels=digits.target_names
).plot(
    ax=axes[0],
    cmap="Blues",
    colorbar=False
)

axes[0].set_title("DBN + Logistic Regression")

ConfusionMatrixDisplay(
    confusion_matrix=raw_cm,
    display_labels=digits.target_names
).plot(
    ax=axes[1],
    cmap="Greens",
    colorbar=False
)

axes[1].set_title("Raw Logistic Regression")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 14. VISUALIZE ORIGINAL DIGIT IMAGES
# ------------------------------------------------------------

print_section("13. VISUALIZING ORIGINAL DIGIT IMAGES")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(
        digits.images[i],
        cmap="gray"
    )
    ax.set_title(f"Digit: {y[i]}")
    ax.axis("off")

fig.suptitle("Sample Handwritten Digit Images")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "original_digit_images.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 15. VISUALIZE FIRST RBM LEARNED FEATURES
# ------------------------------------------------------------

print_section("14. VISUALIZING FIRST RBM LEARNED FEATURES")

first_rbm = dbn_model.named_steps["rbm1"]

components = first_rbm.components_

fig, axes = plt.subplots(8, 8, figsize=(10, 10))

for i, ax in enumerate(axes.flat):

    ax.imshow(
        components[i].reshape(8, 8),
        cmap="gray"
    )

    ax.set_title(f"Feature {i + 1}", fontsize=7)
    ax.axis("off")

fig.suptitle("64 Features Learned by the First RBM Layer")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "rbm1_learned_features.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 16. VISUALIZE SECOND RBM FEATURES
# ------------------------------------------------------------

print_section("15. VISUALIZING SECOND RBM FEATURES")

second_rbm = dbn_model.named_steps["rbm2"]

fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axes.flat):

    feature = second_rbm.components_[i]

    ax.bar(
        range(len(feature)),
        feature
    )

    ax.set_title(f"Feature {i + 1}", fontsize=7)
    ax.set_xticks([])

fig.suptitle("32 Features Learned by the Second RBM Layer")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "rbm2_learned_features.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 17. VISUALIZE MODEL ACCURACY COMPARISON
# ------------------------------------------------------------

print_section("16. MODEL ACCURACY COMPARISON")

plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Raw Logistic Regression vs DBN Model")
plt.ylabel("Test Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)

plt.xticks(
    rotation=10,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_accuracy_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 18. VISUALIZE CROSS-VALIDATION ACCURACY
# ------------------------------------------------------------

print_section("17. CROSS-VALIDATION ACCURACY VISUALIZATION")

accuracy_cv = cv_df.pivot(
    index="Fold",
    columns="Model",
    values="Accuracy"
)

accuracy_cv.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("5-Fold Cross-Validation Accuracy Comparison")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cross_validation_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 19. FINAL RESULT SUMMARY
# ------------------------------------------------------------

print_section("18. FINAL RESULT SUMMARY")

print("Dataset: Scikit-learn Digits")
print("Total samples:", len(X))
print("Input features:", X.shape[1])
print("Classes:", len(np.unique(y)))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nBaseline Test Accuracy: {:.4f}".format(
    raw_metrics["Accuracy"]
))

print("DBN Test Accuracy: {:.4f}".format(
    dbn_metrics["Accuracy"]
))

print("\nDBN Cross-Validation Accuracy:")

dbn_cv_accuracy = cv_summary_df[
    (cv_summary_df["Model"] == "DBN Features + Logistic Regression")
    & (cv_summary_df["Metric"] == "accuracy")
]

print(
    "Mean: {:.4f}".format(
        dbn_cv_accuracy["Mean"].iloc[0]
    )
)

print(
    "Standard Deviation: {:.4f}".format(
        dbn_cv_accuracy["Standard Deviation"].iloc[0]
    )
)

print("\nAll result files saved in:", OUTPUT_DIR)

print_section("PRACTICAL 6 EXECUTION COMPLETED SUCCESSFULLY")
