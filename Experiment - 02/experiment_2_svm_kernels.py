from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUT_DIR / "experiment_output.txt"


def main():
    report_lines = []

    def log(message=""):
        message = str(message)
        print(message)
        report_lines.append(message)

    log("=" * 72)
    log("EXPERIMENT 2: SVM CLASSIFIER WITH DIFFERENT KERNELS")
    log("=" * 72)

    # 1. Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    df = pd.DataFrame(X, columns=iris.feature_names)
    df["target"] = y
    df["target_name"] = [iris.target_names[index] for index in y]

    log("\n1. DATASET INFORMATION")
    log("-" * 40)
    log(f"Dataset: Iris")
    log(f"Number of samples: {X.shape[0]}")
    log(f"Number of features: {X.shape[1]}")
    log(f"Target classes: {', '.join(iris.target_names)}")
    log("\nFirst five rows:")
    log(df.head().to_string(index=False))
    log("\nClass distribution:")
    log(df["target_name"].value_counts().to_string())

    df.to_csv(OUTPUT_DIR / "iris_dataset.csv", index=False)

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    log("\n2. TRAIN-TEST SPLIT")
    log("-" * 40)
    log(f"Training samples: {len(X_train)}")
    log(f"Testing samples: {len(X_test)}")
    log("Test size: 20%")
    log("Random state: 42")
    log("Stratified split: Yes")

    # 3. Standardize the features using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log("\n3. FEATURE SCALING")
    log("-" * 40)
    log("StandardScaler fitted on training data and applied to test data.")

    # 4. Train and evaluate each SVM kernel
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    results = []
    predictions = {}
    confusion_matrices = {}

    log("\n4. MODEL TRAINING AND EVALUATION")
    log("-" * 40)

    for kernel in kernels:
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=iris.target_names,
            zero_division=0,
        )

        predictions[kernel] = y_pred
        confusion_matrices[kernel] = cm

        results.append({
            "Kernel": kernel,
            "Accuracy": accuracy,
            "Accuracy_Percentage": accuracy * 100,
            "Correct_Predictions": int(np.sum(y_test == y_pred)),
            "Incorrect_Predictions": int(np.sum(y_test != y_pred)),
            "Support_Vectors_Total": int(np.sum(model.n_support_)),
        })

        log(f"\nKernel: {kernel}")
        log(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        log(f"Correct predictions: {int(np.sum(y_test == y_pred))}/{len(y_test)}")
        log("Confusion Matrix:")
        log(cm)
        log("Classification Report:")
        log(report)

    # 5. Save comparison table
    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy", ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "kernel_accuracy_comparison.csv", index=False)

    log("\n5. ACCURACY COMPARISON")
    log("-" * 40)
    log(results_df.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Accuracy_Percentage": "{:.2f}".format,
    }))

    best_accuracy = results_df["Accuracy"].max()
    best_kernels = results_df.loc[
        np.isclose(results_df["Accuracy"], best_accuracy), "Kernel"
    ].tolist()

    log(f"\nHighest test accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    log(f"Best-performing kernel(s): {', '.join(best_kernels)}")

    # Save per-kernel classification reports and confusion matrices
    detail_lines = [
        "EXPERIMENT 2: PER-KERNEL CLASSIFICATION DETAILS",
        "=" * 60,
    ]

    for kernel in kernels:
        detail_lines.extend([
            f"\nKERNEL: {kernel}",
            "-" * 40,
            "Confusion Matrix:",
            str(confusion_matrices[kernel]),
            "Classification Report:",
            classification_report(
                y_test,
                predictions[kernel],
                labels=[0, 1, 2],
                target_names=iris.target_names,
                zero_division=0,
            ),
        ])

    (OUTPUT_DIR / "classification_details.txt").write_text(
        "\n".join(detail_lines) + "\n",
        encoding="utf-8",
    )

    # 6. Graph: kernel accuracy comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        results_df["Kernel"],
        results_df["Accuracy"],
        edgecolor="black",
    )
    plt.xlabel("SVM Kernel")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy Comparison of SVM Kernels")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, accuracy in zip(bars, results_df["Accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kernel_accuracy_comparison.png", dpi=160)
    plt.close()

    # 7. Graph: confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, kernel in zip(axes.flat, kernels):
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrices[kernel],
            display_labels=iris.target_names,
        )
        display.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{kernel.title()} Kernel")

    fig.suptitle("Confusion Matrices for Different SVM Kernels", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    # 8. Graph: classification accuracy as percentages
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        results_df["Kernel"],
        results_df["Accuracy_Percentage"],
        edgecolor="black",
    )
    plt.xlabel("SVM Kernel")
    plt.ylabel("Accuracy (%)")
    plt.title("SVM Kernel Classification Accuracy")
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, accuracy in zip(bars, results_df["Accuracy_Percentage"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kernel_accuracy_percentage.png", dpi=160)
    plt.close()

    # 9. Graph: dataset feature distributions by class
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flat

    for index, feature_name in enumerate(iris.feature_names):
        ax = axes[index]
        for class_index, class_name in enumerate(iris.target_names):
            ax.hist(
                X[y == class_index, index],
                bins=12,
                alpha=0.55,
                label=class_name,
            )
        ax.set_title(feature_name)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.suptitle("Iris Feature Distributions by Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "iris_feature_distributions.png", dpi=160)
    plt.close(fig)

    # 10. Final result and report
    log("\n6. GENERATED OUTPUT FILES")
    log("-" * 40)

    output_files = sorted(
        path.name for path in OUTPUT_DIR.iterdir() if path.is_file()
        and path.name != REPORT_PATH.name
    )

    for filename in output_files:
        log(f"- {filename}")

    log("\n7. RESULT")
    log("-" * 40)
    log("SVM classifiers were trained using linear, polynomial, RBF, and sigmoid kernels.")
    log(f"The highest observed test accuracy was {best_accuracy * 100:.2f}%.")
    log("The results show how kernel selection can affect classification performance.")
    log("Feature scaling was applied before model training.")
    log("\nExperiment completed successfully.")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\nFull report saved to: {REPORT_PATH}")
    print(f"All output files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
