"""
============================================================
PRACTICAL 7: GIBBS SAMPLING
Course Outcome: CO4
============================================================

Aim:
Implement Gibbs Sampling in Python and generate samples
from a bivariate normal probability distribution.

Distribution:
    Mean X = 0
    Mean Y = 0
    Standard Deviation X = 1
    Standard Deviation Y = 1
    Correlation = 0.8

Sampling configuration:
    Total iterations = 10000
    Burn-in iterations = 1000
    Random seed = 42

Generated outputs:
    CSV files containing samples and statistical summaries.
    Five visualization images.
    Complete experiment output log.
============================================================
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_SEED = 42

MU_X = 0.0
MU_Y = 0.0

SIGMA_X = 1.0
SIGMA_Y = 1.0

RHO = 0.8

NUM_SAMPLES = 10000
BURN_IN = 1000

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ============================================================
# 2. VALIDATE DISTRIBUTION PARAMETERS
# ============================================================

def validate_parameters():
    if NUM_SAMPLES <= 0:
        raise ValueError("Number of samples must be positive.")

    if BURN_IN < 0 or BURN_IN >= NUM_SAMPLES:
        raise ValueError("Burn-in must be between 0 and NUM_SAMPLES.")

    if SIGMA_X <= 0 or SIGMA_Y <= 0:
        raise ValueError("Standard deviations must be positive.")

    if not -1.0 < RHO < 1.0:
        raise ValueError("Correlation must be strictly between -1 and 1.")


# ============================================================
# 3. GIBBS SAMPLING IMPLEMENTATION
# ============================================================

def gibbs_sampling():
    """
    Generate samples from a bivariate normal distribution
    using sequential conditional sampling.

    Conditional distribution X | Y:

    Mean = mu_x + rho * (sigma_x / sigma_y) * (Y - mu_y)

    Standard deviation = sigma_x * sqrt(1 - rho^2)

    Conditional distribution Y | X:

    Mean = mu_y + rho * (sigma_y / sigma_x) * (X - mu_x)

    Standard deviation = sigma_y * sqrt(1 - rho^2)
    """

    x_samples = np.zeros(NUM_SAMPLES, dtype=float)
    y_samples = np.zeros(NUM_SAMPLES, dtype=float)

    # Initial values
    x_samples[0] = 0.0
    y_samples[0] = 0.0

    conditional_std_x = SIGMA_X * np.sqrt(1 - RHO**2)
    conditional_std_y = SIGMA_Y * np.sqrt(1 - RHO**2)

    for i in range(1, NUM_SAMPLES):

        # Sample X conditioned on the previous Y
        mean_x_given_y = (
            MU_X
            + RHO * (SIGMA_X / SIGMA_Y)
            * (y_samples[i - 1] - MU_Y)
        )

        x_samples[i] = np.random.normal(
            mean_x_given_y,
            conditional_std_x
        )

        # Sample Y conditioned on the updated X
        mean_y_given_x = (
            MU_Y
            + RHO * (SIGMA_Y / SIGMA_X)
            * (x_samples[i] - MU_X)
        )

        y_samples[i] = np.random.normal(
            mean_y_given_x,
            conditional_std_y
        )

    return x_samples, y_samples


# ============================================================
# 4. STATISTICAL ANALYSIS
# ============================================================

def calculate_statistics(x_final, y_final):

    sample_correlation = np.corrcoef(x_final, y_final)[0, 1]

    statistics = {
        "Parameter": [
            "Mean of X",
            "Mean of Y",
            "Standard Deviation of X",
            "Standard Deviation of Y",
            "Correlation between X and Y",
        ],
        "Theoretical Value": [
            MU_X,
            MU_Y,
            SIGMA_X,
            SIGMA_Y,
            RHO,
        ],
        "Observed Value": [
            np.mean(x_final),
            np.mean(y_final),
            np.std(x_final, ddof=1),
            np.std(y_final, ddof=1),
            sample_correlation,
        ],
    }

    statistics_df = pd.DataFrame(statistics)

    statistics_df["Absolute Difference"] = (
        statistics_df["Observed Value"]
        - statistics_df["Theoretical Value"]
    ).abs()

    return statistics_df


# ============================================================
# 5. VISUALIZATION
# ============================================================

def save_visualizations(x_samples, y_samples, x_final, y_final):

    # --------------------------------------------------------
    # Plot 1: Scatter Plot
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x_final,
        y_final,
        alpha=0.3,
        s=10
    )

    plt.xlabel("X Samples")
    plt.ylabel("Y Samples")

    plt.title(
        "Gibbs Sampling: Bivariate Normal Distribution"
    )

    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gibbs_scatter_plot.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: Histogram of X
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        x_final,
        bins=40,
        density=True,
        alpha=0.7,
        edgecolor="black"
    )

    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Histogram of X Samples")

    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gibbs_histogram_x.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: Histogram of Y
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        y_final,
        bins=40,
        density=True,
        alpha=0.7,
        edgecolor="black"
    )

    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.title("Histogram of Y Samples")

    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gibbs_histogram_y.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 4: Trace Plot of X
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        np.arange(1000),
        x_samples[:1000],
        linewidth=0.8
    )

    plt.xlabel("Iteration")
    plt.ylabel("X Value")
    plt.title("Trace Plot of X Samples")

    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gibbs_trace_x.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 5: Trace Plot of Y
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        np.arange(1000),
        y_samples[:1000],
        linewidth=0.8
    )

    plt.xlabel("Iteration")
    plt.ylabel("Y Value")
    plt.title("Trace Plot of Y Samples")

    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gibbs_trace_y.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("PRACTICAL 7: GIBBS SAMPLING")
    print("COURSE OUTCOME: CO4")
    print("=" * 65)

    validate_parameters()

    print("\n1. EXPERIMENT CONFIGURATION")
    print("-" * 65)

    print(f"Random seed             : {RANDOM_SEED}")
    print(f"Mean of X               : {MU_X}")
    print(f"Mean of Y               : {MU_Y}")
    print(f"Standard deviation of X : {SIGMA_X}")
    print(f"Standard deviation of Y : {SIGMA_Y}")
    print(f"Correlation             : {RHO}")
    print(f"Total iterations        : {NUM_SAMPLES}")
    print(f"Burn-in iterations      : {BURN_IN}")

    # --------------------------------------------------------
    # Generate samples
    # --------------------------------------------------------

    print("\n2. RUNNING GIBBS SAMPLING")
    print("-" * 65)

    x_samples, y_samples = gibbs_sampling()

    print("[SUCCESS] Gibbs Sampling completed.")

    # --------------------------------------------------------
    # Remove burn-in samples
    # --------------------------------------------------------

    x_final = x_samples[BURN_IN:]
    y_final = y_samples[BURN_IN:]

    print("\n3. BURN-IN REMOVAL")
    print("-" * 65)

    print(f"Total generated samples : {len(x_samples)}")
    print(f"Burn-in samples removed : {BURN_IN}")
    print(f"Final retained samples  : {len(x_final)}")

    # --------------------------------------------------------
    # Create DataFrames
    # --------------------------------------------------------

    all_samples_df = pd.DataFrame({
        "Iteration": np.arange(NUM_SAMPLES),
        "X": x_samples,
        "Y": y_samples,
        "Burn_In": np.arange(NUM_SAMPLES) < BURN_IN,
    })

    final_samples_df = pd.DataFrame({
        "Iteration": np.arange(BURN_IN, NUM_SAMPLES),
        "X": x_final,
        "Y": y_final,
    })

    all_samples_df.to_csv(
        OUTPUT_DIR / "all_gibbs_samples.csv",
        index=False
    )

    final_samples_df.to_csv(
        OUTPUT_DIR / "gibbs_samples_after_burnin.csv",
        index=False
    )

    # --------------------------------------------------------
    # Display generated samples
    # --------------------------------------------------------

    print("\n4. FIRST FIVE RETAINED SAMPLES")
    print("-" * 65)

    print(final_samples_df[["X", "Y"]].head().to_string(index=False))

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    print("\n5. STATISTICAL ANALYSIS")
    print("-" * 65)

    statistics_df = calculate_statistics(x_final, y_final)

    print(statistics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    statistics_df.to_csv(
        OUTPUT_DIR / "gibbs_statistical_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save correlation matrix
    # --------------------------------------------------------

    correlation_matrix = final_samples_df[["X", "Y"]].corr()

    correlation_matrix.to_csv(
        OUTPUT_DIR / "gibbs_correlation_matrix.csv"
    )

    print("\n6. CORRELATION MATRIX")
    print("-" * 65)

    print(correlation_matrix.to_string(float_format=lambda x: f"{x:.6f}"))

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    print("\n7. GENERATING VISUALIZATIONS")
    print("-" * 65)

    save_visualizations(
        x_samples,
        y_samples,
        x_final,
        y_final
    )

    print("[SUCCESS] All five visualizations saved.")

    # --------------------------------------------------------
    # Final result summary
    # --------------------------------------------------------

    print("\n8. EXPERIMENT SUMMARY")
    print("-" * 65)

    observed_mean_x = np.mean(x_final)
    observed_mean_y = np.mean(y_final)
    observed_corr = np.corrcoef(x_final, y_final)[0, 1]

    print(f"Observed mean of X       : {observed_mean_x:.6f}")
    print(f"Observed mean of Y       : {observed_mean_y:.6f}")
    print(f"Observed correlation     : {observed_corr:.6f}")

    print(f"Theoretical mean of X    : {MU_X:.6f}")
    print(f"Theoretical mean of Y    : {MU_Y:.6f}")
    print(f"Theoretical correlation  : {RHO:.6f}")

    print("\n9. GENERATED OUTPUT FILES")
    print("-" * 65)

    for output_file in sorted(OUTPUT_DIR.iterdir()):
        if output_file.is_file():
            print(
                f"{output_file.name:<42} "
                f"{output_file.stat().st_size:>10} bytes"
            )

    print("\n" + "=" * 65)
    print("PRACTICAL 7 EXECUTION COMPLETED SUCCESSFULLY")
    print("Gibbs Sampling generated samples from the target distribution.")
    print("END OF EXPERIMENT")
    print("=" * 65)


# ============================================================
# 7. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\n[ERROR] Experiment execution failed.")
        print(f"Reason: {error}")
        traceback.print_exc()
        sys.exit(1)
