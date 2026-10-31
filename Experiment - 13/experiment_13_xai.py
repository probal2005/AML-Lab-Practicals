"""
PRACTICAL 13: EXPLAINABLE AI USING SHAP AND LIME

Dataset:
    Breast Cancer Wisconsin Diagnostic Dataset

Model:
    Random Forest Classifier

Explainability:
    SHAP (SHapley Additive exPlanations)
    LIME (Local Interpretable Model-agnostic Explanations)
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
EXPLANATION_DIR = OUTPUT_DIR / "explanations"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_TO_EXPLAIN = 1
LIME_INSTANCE_INDEX = 0


def create_directories() -> None:
    """Create all required output directories."""
    for directory in (
        MODEL_DIR,
        PLOT_DIR,
        LOG_DIR,
        EXPLANATION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    """Configure console and file logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(
                LOG_DIR / "training.log",
                mode="w",
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_dataset():
    """Load the built-in Breast Cancer Wisconsin dataset."""
    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names,
    )

    y = pd.Series(
        dataset.target,
        name="target",
    )

    class_names = ["malignant", "benign"]

    logging.info("Dataset loaded successfully.")
    logging.info("Total samples: %d", X.shape[0])
    logging.info("Total features: %d", X.shape[1])
    logging.info("Class distribution: %s", y.value_counts().sort_index().to_dict())

    return X, y, class_names


def train_model(X, y):
    """Split the dataset and train the Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE,
    )

    logging.info("Training Random Forest classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["malignant", "benign"],
        digits=4,
    )

    logging.info("Model training completed.")
    logging.info("Training samples: %d", len(X_train))
    logging.info("Testing samples: %d", len(X_test))
    logging.info("Accuracy: %.6f", accuracy)
    logging.info("Confusion matrix:\n%s", matrix)
    logging.info("Classification report:\n%s", report)

    joblib.dump(
        model,
        MODEL_DIR / "random_forest.joblib",
    )

    (LOG_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    metrics = {
        "dataset": "Breast Cancer Wisconsin Diagnostic Dataset",
        "total_samples": int(len(X)),
        "total_features": int(X.shape[1]),
        "class_names": ["malignant", "benign"],
        "class_distribution": {
            "malignant": int((y == 0).sum()),
            "benign": int((y == 1).sum()),
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 5,
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["malignant", "benign"],
            output_dict=True,
            zero_division=0,
        ),
    }

    (LOG_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        probabilities,
        accuracy,
        matrix,
    )


def create_confusion_matrix_plot(matrix, class_names) -> None:
    """Save the confusion matrix visualization."""
    figure, axis = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title("Random Forest - Confusion Matrix")
    figure.tight_layout()

    figure.savefig(
        PLOT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def explain_with_shap(model, X_train, X_test) -> None:
    """Generate global and local SHAP explanations."""
    logging.info("Starting SHAP explanations...")

    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=RANDOM_STATE,
    )

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )

    explanation_data = X_test.copy()

    shap_values = explainer(explanation_data)

    # For a binary classifier, explain class 1 (benign).
    if shap_values.values.ndim == 3:
        class_explanation = shap.Explanation(
            values=shap_values.values[:, :, CLASS_TO_EXPLAIN],
            base_values=shap_values.base_values[:, CLASS_TO_EXPLAIN],
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    else:
        class_explanation = shap_values

    # Global SHAP summary plot.
    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        class_explanation.values,
        explanation_data,
        feature_names=list(X_test.columns),
        show=False,
        max_display=15,
    )

    plt.title("SHAP Summary Plot - Benign Class")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Global SHAP feature importance plot.
    plt.figure(figsize=(11, 8))

    shap.summary_plot(
        class_explanation.values,
        explanation_data,
        feature_names=list(X_test.columns),
        plot_type="bar",
        show=False,
        max_display=15,
    )

    plt.title("SHAP Feature Importance - Benign Class")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Local SHAP waterfall explanation for the first test sample.
    local_explanation = shap.Explanation(
        values=class_explanation.values[0],
        base_values=class_explanation.base_values[0],
        data=class_explanation.data[0],
        feature_names=list(X_test.columns),
    )

    plt.figure(figsize=(11, 8))

    shap.plots.waterfall(
        local_explanation,
        max_display=15,
        show=False,
    )

    plt.title("SHAP Local Waterfall - Benign Class")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logging.info("SHAP explanations saved successfully.")


def explain_with_lime(
    model,
    X_train,
    X_test,
    y_test,
    class_names,
) -> dict:
    """Generate a local LIME explanation for one test instance."""
    logging.info("Starting LIME explanation...")

    lime_explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    instance = X_test.iloc[LIME_INSTANCE_INDEX].to_numpy()

    explanation = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=10,
        num_samples=5000,
        labels=(CLASS_TO_EXPLAIN,),
    )

    predicted_class = int(model.predict(instance.reshape(1, -1))[0])
    predicted_probabilities = model.predict_proba(
        instance.reshape(1, -1)
    )[0]

    actual_class = int(y_test.iloc[LIME_INSTANCE_INDEX])

    explanation.save_to_file(
        str(EXPLANATION_DIR / "lime_explanation.html")
    )

    figure = explanation.as_pyplot_figure(
        label=CLASS_TO_EXPLAIN
    )

    figure.set_size_inches(11, 7)
    figure.tight_layout()

    figure.savefig(
        PLOT_DIR / "lime_explanation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    lime_features = [
        {
            "feature_rule": str(feature),
            "contribution": float(weight),
        }
        for feature, weight in explanation.as_list(
            label=CLASS_TO_EXPLAIN
        )
    ]

    result = {
        "test_sample_index": LIME_INSTANCE_INDEX,
        "actual_class": class_names[actual_class],
        "predicted_class": class_names[predicted_class],
        "predicted_probabilities": {
            class_names[index]: float(probability)
            for index, probability in enumerate(
                predicted_probabilities
            )
        },
        "explained_class": class_names[CLASS_TO_EXPLAIN],
        "lime_features": lime_features,
    }

    (EXPLANATION_DIR / "lime_explanation.json").write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )

    logging.info("LIME actual class: %s", class_names[actual_class])
    logging.info("LIME predicted class: %s", class_names[predicted_class])
    logging.info("LIME explanation saved successfully.")

    return result


def verify_outputs() -> None:
    """Verify that all expected output files exist and are non-empty."""
    expected_files = [
        MODEL_DIR / "random_forest.joblib",
        LOG_DIR / "classification_report.txt",
        LOG_DIR / "metrics.json",
        LOG_DIR / "training.log",
        PLOT_DIR / "confusion_matrix.png",
        PLOT_DIR / "shap_summary.png",
        PLOT_DIR / "shap_feature_importance.png",
        PLOT_DIR / "shap_waterfall.png",
        PLOT_DIR / "lime_explanation.png",
        EXPLANATION_DIR / "lime_explanation.html",
        EXPLANATION_DIR / "lime_explanation.json",
    ]

    missing_files = [
        path for path in expected_files
        if not path.is_file() or path.stat().st_size == 0
    ]

    if missing_files:
        raise RuntimeError(
            "Output verification failed: "
            + ", ".join(str(path) for path in missing_files)
        )

    logging.info("All %d expected output files verified.", len(expected_files))


def main() -> None:
    create_directories()
    configure_logging()

    logging.info("=" * 65)
    logging.info("PRACTICAL 13: EXPLAINABLE AI USING SHAP AND LIME")
    logging.info("=" * 65)

    X, y, class_names = load_dataset()

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        probabilities,
        accuracy,
        matrix,
    ) = train_model(X, y)

    create_confusion_matrix_plot(matrix, class_names)

    explain_with_shap(
        model,
        X_train,
        X_test,
    )

    lime_result = explain_with_lime(
        model,
        X_train,
        X_test,
        y_test,
        class_names,
    )

    verify_outputs()

    logging.info("=" * 65)
    logging.info("PRACTICAL 13 COMPLETED SUCCESSFULLY")
    logging.info("Final test accuracy: %.4f", accuracy)
    logging.info("LIME explained sample: %d", LIME_INSTANCE_INDEX)
    logging.info("LIME actual class: %s", lime_result["actual_class"])
    logging.info("LIME predicted class: %s", lime_result["predicted_class"])
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
