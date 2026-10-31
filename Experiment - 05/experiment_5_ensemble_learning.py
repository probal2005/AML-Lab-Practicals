"""
PRACTICAL 5: Ensemble Learning with Cross-Validation
and Hyperparameter Analysis

Course Outcome: CO3
Dataset: Breast Cancer Wisconsin Dataset
"""

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5


def save_figure(filename):
    """Save the current Matplotlib figure."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    start_time = time.time()

    print("=" * 75)
    print("PRACTICAL 5: ENSEMBLE LEARNING")
    print("CROSS-VALIDATION AND HYPERPARAMETER ANALYSIS")
    print("=" * 75)

    print("\n1. ENVIRONMENT INFORMATION")
    print("-" * 75)
    print("Python version:", sys.version.split()[0])
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)

    # ---------------------------------------------------------
    # 2. LOAD DATASET
    # ---------------------------------------------------------
    print("\n2. LOADING BREAST CANCER WISCONSIN DATASET")
    print("-" * 75)

    data = load_breast_cancer()

    X = data.data
    y = data.target

    df = pd.DataFrame(X, columns=data.feature_names)
    df["target"] = y
    df["diagnosis"] = pd.Series(y).map(
        {0: data.target_names[0], 1: data.target_names[1]}
    )

    df.to_csv(OUTPUT_DIR / "breast_cancer_dataset.csv", index=False)

    print("First five records:")
    print(df.head().to_string())

    print("\nDataset shape:", df.shape)
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Target classes:", list(data.target_names))

    print("\nClass distribution:")
    print(df["diagnosis"].value_counts().to_string())

    print("\nMissing values:", int(df.isnull().sum().sum()))

    # ---------------------------------------------------------
    # 3. TRAIN-TEST SPLIT
    # ---------------------------------------------------------
    print("\n3. TRAIN-TEST SPLIT")
    print("-" * 75)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # These tree-based ensemble models do not require scaling.
    print(
        "Feature scaling: Not required for the selected "
        "tree-based ensemble models."
    )

    # ---------------------------------------------------------
    # 4. DEFINE ENSEMBLE MODELS
    # ---------------------------------------------------------
    print("\n4. DEFINING ENSEMBLE MODELS")
    print("-" * 75)

    models = {
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }

    for name, model in models.items():
        print(f"{name}: {model.__class__.__name__}")

    # ---------------------------------------------------------
    # 5. FIVE-FOLD CROSS-VALIDATION
    # ---------------------------------------------------------
    print("\n5. FIVE-FOLD CROSS-VALIDATION")
    print("-" * 75)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = []

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        cv_results.append(
            {
                "Model": name,
                "Mean CV Accuracy": scores.mean(),
                "Standard Deviation": scores.std(),
                "Minimum CV Accuracy": scores.min(),
                "Maximum CV Accuracy": scores.max(),
            }
        )

        print(f"\n{name}")
        print("Fold scores:", np.round(scores, 4))
        print(f"Mean accuracy: {scores.mean():.4f}")
        print(f"Standard deviation: {scores.std():.4f}")

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(
        OUTPUT_DIR / "cross_validation_results.csv",
        index=False,
    )

    print("\nCross-validation comparison:")
    print(cv_results_df.to_string(index=False))

    # ---------------------------------------------------------
    # 6. BASELINE MODEL TEST EVALUATION
    # ---------------------------------------------------------
    print("\n6. BASELINE MODEL TEST EVALUATION")
    print("-" * 75)

    baseline_results = []
    baseline_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        baseline_models[name] = model

        baseline_results.append(
            {
                "Model": name,
                "Test Accuracy": accuracy,
            }
        )

        print("\n" + "=" * 65)
        print("MODEL:", name)
        print("=" * 65)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=data.target_names,
                zero_division=0,
            )
        )

        report = classification_report(
            y_test,
            y_pred,
            target_names=data.target_names,
            output_dict=True,
            zero_division=0,
        )

        pd.DataFrame(report).transpose().to_csv(
            OUTPUT_DIR / (
                name.lower().replace(" ", "_")
                + "_classification_report.csv"
            )
        )

        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=data.target_names,
            cmap="Blues",
            values_format="d",
        )
        display.ax_.set_title(f"{name} - Confusion Matrix")

        save_figure(
            name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )

    baseline_results_df = pd.DataFrame(baseline_results)

    baseline_results_df.to_csv(
        OUTPUT_DIR / "baseline_test_results.csv",
        index=False,
    )

    print("\nBaseline test accuracy:")
    print(baseline_results_df.to_string(index=False))

    # ---------------------------------------------------------
    # 7. RANDOM FOREST HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    print("\n7. RANDOM FOREST HYPERPARAMETER TUNING")
    print("-" * 75)

    rf_param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
    }

    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid=rf_param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )

    rf_grid.fit(X_train, y_train)

    print("Best parameters:", rf_grid.best_params_)
    print(f"Best CV accuracy: {rf_grid.best_score_:.4f}")

    # ---------------------------------------------------------
    # 8. ADABOOST HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    print("\n8. ADABOOST HYPERPARAMETER TUNING")
    print("-" * 75)

    ada_param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    }

    ada_grid = GridSearchCV(
        estimator=AdaBoostClassifier(
            random_state=RANDOM_STATE,
        ),
        param_grid=ada_param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )

    ada_grid.fit(X_train, y_train)

    print("Best parameters:", ada_grid.best_params_)
    print(f"Best CV accuracy: {ada_grid.best_score_:.4f}")

    # ---------------------------------------------------------
    # 9. GRADIENT BOOSTING HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    print("\n9. GRADIENT BOOSTING HYPERPARAMETER TUNING")
    print("-" * 75)

    gb_param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.5],
        "max_depth": [2, 3, 5],
    }

    gb_grid = GridSearchCV(
        estimator=GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        param_grid=gb_param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )

    gb_grid.fit(X_train, y_train)

    print("Best parameters:", gb_grid.best_params_)
    print(f"Best CV accuracy: {gb_grid.best_score_:.4f}")

    # ---------------------------------------------------------
    # 10. SAVE HYPERPARAMETER SEARCH RESULTS
    # ---------------------------------------------------------
    print("\n10. SAVING HYPERPARAMETER SEARCH RESULTS")
    print("-" * 75)

    grids = {
        "Random Forest": rf_grid,
        "AdaBoost": ada_grid,
        "Gradient Boosting": gb_grid,
    }

    tuning_summary = []

    for name, grid in grids.items():
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv(
            OUTPUT_DIR / (
                name.lower().replace(" ", "_")
                + "_grid_search_results.csv"
            ),
            index=False,
        )

        tuning_summary.append(
            {
                "Model": name,
                "Best Parameters": str(grid.best_params_),
                "Best CV Accuracy": grid.best_score_,
                "Number of Combinations": len(grid.cv_results_),
            }
        )

    tuning_summary_df = pd.DataFrame(tuning_summary)
    tuning_summary_df.to_csv(
        OUTPUT_DIR / "hyperparameter_tuning_summary.csv",
        index=False,
    )

    print(tuning_summary_df.to_string(index=False))

    # ---------------------------------------------------------
    # 11. EVALUATE TUNED MODELS
    # ---------------------------------------------------------
    print("\n11. TUNED MODEL TEST EVALUATION")
    print("-" * 75)

    tuned_models = {
        "Tuned Random Forest": rf_grid.best_estimator_,
        "Tuned AdaBoost": ada_grid.best_estimator_,
        "Tuned Gradient Boosting": gb_grid.best_estimator_,
    }

    tuned_results = []

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        tuned_results.append(
            {
                "Model": name,
                "Test Accuracy": accuracy,
            }
        )

        print(f"\n{name}")
        print(f"Test accuracy: {accuracy:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=data.target_names,
                zero_division=0,
            )
        )

        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=data.target_names,
            cmap="Greens",
            values_format="d",
        )
        display.ax_.set_title(f"{name} - Confusion Matrix")

        save_figure(
            name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )

    tuned_results_df = pd.DataFrame(tuned_results)
    tuned_results_df.to_csv(
        OUTPUT_DIR / "tuned_model_test_results.csv",
        index=False,
    )

    print("\nTuned model comparison:")
    print(tuned_results_df.to_string(index=False))

    # ---------------------------------------------------------
    # 12. CROSS-VALIDATION ACCURACY GRAPH
    # ---------------------------------------------------------
    print("\n12. GENERATING COMPARISON GRAPHS")
    print("-" * 75)

    plt.figure(figsize=(9, 5))
    plt.bar(
        cv_results_df["Model"],
        cv_results_df["Mean CV Accuracy"],
    )
    plt.xlabel("Ensemble Model")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title("Five-Fold Cross-Validation Accuracy")
    plt.ylim(0.85, 1.0)
    plt.xticks(rotation=15)
    save_figure("cross_validation_accuracy.png")

    # ---------------------------------------------------------
    # 13. TUNED MODEL ACCURACY GRAPH
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.bar(
        tuned_results_df["Model"],
        tuned_results_df["Test Accuracy"],
    )
    plt.xlabel("Tuned Ensemble Model")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy After Hyperparameter Tuning")
    plt.ylim(0.85, 1.0)
    plt.xticks(rotation=15)
    save_figure("tuned_model_test_accuracy.png")

    # ---------------------------------------------------------
    # 14. HYPERPARAMETER COMPARISON GRAPH
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.bar(
        tuning_summary_df["Model"],
        tuning_summary_df["Best CV Accuracy"],
    )
    plt.xlabel("Ensemble Model")
    plt.ylabel("Best Cross-Validation Accuracy")
    plt.title("Best Accuracy After Hyperparameter Tuning")
    plt.ylim(0.85, 1.0)
    plt.xticks(rotation=15)
    save_figure("best_hyperparameter_cv_accuracy.png")

    # ---------------------------------------------------------
    # 15. COMBINED BASELINE AND TUNED COMPARISON
    # ---------------------------------------------------------
    combined = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "AdaBoost",
                "Gradient Boosting",
            ],
            "Baseline Test Accuracy": baseline_results_df[
                "Test Accuracy"
            ].values,
            "Tuned Test Accuracy": tuned_results_df[
                "Test Accuracy"
            ].values,
        }
    )

    combined.to_csv(
        OUTPUT_DIR / "baseline_vs_tuned_comparison.csv",
        index=False,
    )

    combined_plot = combined.set_index("Model")
    combined_plot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.xlabel("Ensemble Model")
    plt.ylabel("Test Accuracy")
    plt.title("Baseline vs Tuned Ensemble Models")
    plt.ylim(0.85, 1.0)
    plt.xticks(rotation=15)
    plt.legend(title="Evaluation")
    save_figure("baseline_vs_tuned_accuracy.png")

    # ---------------------------------------------------------
    # 16. FINAL RESULT
    # ---------------------------------------------------------
    print("\n16. FINAL RESULT")
    print("-" * 75)

    best_index = tuned_results_df["Test Accuracy"].idxmax()
    best_model = tuned_results_df.loc[best_index]

    print("Best-performing tuned model:", best_model["Model"])
    print(f"Test accuracy: {best_model['Test Accuracy']:.4f}")

    print("\nImportant interpretation:")
    print(
        "The best model is selected using the highest test accuracy "
        "among the tuned models."
    )
    print(
        "Cross-validation scores estimate performance during model "
        "selection, while the held-out test set provides the final "
        "evaluation."
    )

    elapsed = time.time() - start_time

    print(f"\nTotal execution time: {elapsed:.2f} seconds")

    print("\nGenerated output files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size:,} bytes)")

    print("\n" + "=" * 75)
    print("PRACTICAL 5 EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
