# Practical 1: Linear Regression vs Ridge Regression vs Lasso Regression

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Capture the actual results in a text report as well as printing them.
report_lines = []


def log(message=""):
    print(message)
    report_lines.append(str(message))


def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(y_true, y_pred),
    }


def main():
    log("=" * 72)
    log("PRACTICAL 1: LINEAR, RIDGE AND LASSO REGRESSION")
    log("=" * 72)
    log(f"Python version: {sys.version.split()[0]}")

    # 1. Load the Diabetes dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    log("\n1. DATASET INFORMATION")
    log(f"Number of samples: {X.shape[0]}")
    log(f"Number of features: {X.shape[1]}")
    log(f"Features: {', '.join(diabetes.feature_names)}")
    log("Dataset loaded successfully.")

    # 2. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log("\n2. TRAIN-TEST SPLIT")
    log(f"Training samples: {len(X_train)}")
    log(f"Testing samples: {len(X_test)}")
    log("Test size: 20%; random_state=42")

    # 3. Scale using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log("\n3. FEATURE SCALING")
    log("StandardScaler fitted on training data only.")

    # 4. Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    }

    results = []
    predictions = {}
    fitted_models = {}

    # 5. Train and predict
    log("\n4. MODEL TRAINING")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model
        predictions[name] = model.predict(X_test_scaled)
        results.append(evaluate_model(name, y_test, predictions[name]))
        log(f"{name}: trained successfully.")

    results_df = pd.DataFrame(results)

    log("\n5. MODEL PERFORMANCE COMPARISON")
    log(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    results_df.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)

    # 6. Compare model coefficients
    coefficient_df = pd.DataFrame({
        "Feature": diabetes.feature_names,
        "Linear Regression": fitted_models["Linear Regression"].coef_,
        "Ridge Regression": fitted_models["Ridge Regression"].coef_,
        "Lasso Regression": fitted_models["Lasso Regression"].coef_,
    })

    log("\n6. COEFFICIENT COMPARISON")
    log(coefficient_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    coefficient_df.to_csv(OUTPUT_DIR / "coefficient_comparison.csv", index=False)

    zero_count = int(np.sum(np.isclose(
        fitted_models["Lasso Regression"].coef_, 0.0
    )))
    log("\n7. LASSO FEATURE-SELECTION ANALYSIS")
    log(f"Lasso coefficients equal to zero: {zero_count} of {len(diabetes.feature_names)}")

    # 7. Graph: Actual versus predicted values
    plt.figure(figsize=(9, 6))
    for name, predicted in predictions.items():
        plt.scatter(y_test, predicted, alpha=0.65, label=name)

    lower = min(float(np.min(y_test)), *(float(np.min(p)) for p in predictions.values()))
    upper = max(float(np.max(y_test)), *(float(np.max(p)) for p in predictions.values()))
    plt.plot([lower, upper], [lower, upper], "k--", label="Ideal prediction")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # 8. Graph: R-squared comparison
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["R2 Score"])
    plt.xlabel("Model")
    plt.ylabel("R² Score")
    plt.title("R² Score Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "r2_score_comparison.png", dpi=160)
    plt.close()

    # 9. Graph: MAE and RMSE comparison
    results_df.set_index("Model")[["MAE", "RMSE"]].plot(
        kind="bar", figsize=(9, 5)
    )
    plt.xlabel("Model")
    plt.ylabel("Error (lower is better)")
    plt.title("MAE and RMSE Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance_comparison.png", dpi=160)
    plt.close()

    # 10. Graph: Coefficient comparison
    coefficient_df.set_index("Feature").plot(kind="bar", figsize=(12, 6))
    plt.xlabel("Diabetes Dataset Feature")
    plt.ylabel("Coefficient (standardized features)")
    plt.title("Linear, Ridge and Lasso Coefficient Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coefficient_comparison.png", dpi=160)
    plt.close()

    log("\n8. OUTPUT FILES")
    for filename in [
        "experiment_output.txt",
        "model_performance.csv",
        "coefficient_comparison.csv",
        "actual_vs_predicted.png",
        "r2_score_comparison.png",
        "model_performance_comparison.png",
        "coefficient_comparison.png",
    ]:
        log(f"output/{filename}")

    # Write actual console results into a text file.
    (OUTPUT_DIR / "experiment_output.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    log("\nExperiment completed successfully.")
    # Update the report with the final success message.
    with (OUTPUT_DIR / "experiment_output.txt").open("a", encoding="utf-8") as file:
        file.write("\nExperiment completed successfully.\n")

    print(f"\nReport saved to: {OUTPUT_DIR / 'experiment_output.txt'}")
    print(f"Graphs and CSV files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
