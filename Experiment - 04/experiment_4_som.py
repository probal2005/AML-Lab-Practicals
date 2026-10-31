"""Practical 4: Self-Organizing Map for clustering and visualization."""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_ROWS = 10
GRID_COLUMNS = 10
ITERATIONS = 1000
RANDOM_SEED = 42


def main():
    report = []

    def log(message=""):
        print(message)
        report.append(str(message))

    log("=" * 72)
    log("PRACTICAL 4: SELF-ORGANIZING MAP (SOM)")
    log("=" * 72)

    # 1. Load the Digits dataset.
    digits = load_digits()
    X = digits.data
    y = digits.target

    log("\n1. DATASET INFORMATION")
    log(f"Dataset: Scikit-learn Digits")
    log(f"Dataset shape: {X.shape}")
    log(f"Number of samples: {X.shape[0]}")
    log(f"Number of features: {X.shape[1]}")
    log(f"Number of digit classes: {len(np.unique(y))}")
    log(f"Digit classes: {np.unique(y).tolist()}")

    # Save original feature data.
    feature_columns = [f"pixel_{i}" for i in range(X.shape[1])]
    pd.DataFrame(X, columns=feature_columns).assign(
        actual_digit=y
    ).to_csv(OUTPUT_DIR / "digits_dataset.csv", index=False)

    # 2. Normalize data.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    log("\n2. DATA NORMALIZATION")
    log("Scaler: MinMaxScaler")
    log(f"Normalized feature range: {X_scaled.min():.1f} to {X_scaled.max():.1f}")

    # 3. Initialize and train SOM.
    log("\n3. SOM CONFIGURATION")
    log(f"Grid size: {GRID_ROWS} x {GRID_COLUMNS}")
    log(f"Number of SOM nodes: {GRID_ROWS * GRID_COLUMNS}")
    log("Input dimensions: 64")
    log("Sigma: 1.0")
    log("Initial learning rate: 0.5")
    log(f"Training iterations: {ITERATIONS}")
    log(f"Random seed: {RANDOM_SEED}")

    som = MiniSom(
        x=GRID_ROWS,
        y=GRID_COLUMNS,
        input_len=X_scaled.shape[1],
        sigma=1.0,
        learning_rate=0.5,
        random_seed=RANDOM_SEED,
    )
    som.random_weights_init(X_scaled)
    som.train_random(data=X_scaled, num_iteration=ITERATIONS)

    # 4. Find Best Matching Units.
    bmu_coordinates = np.array([som.winner(sample) for sample in X_scaled])
    cluster_labels = (
        bmu_coordinates[:, 0] * GRID_COLUMNS + bmu_coordinates[:, 1]
    )

    unique_clusters, cluster_counts = np.unique(
        cluster_labels, return_counts=True
    )

    log("\n4. BEST MATCHING UNIT (BMU) RESULTS")
    log("First 10 BMU coordinates:")
    log(np.array2string(bmu_coordinates[:10]))
    log("First 10 cluster labels:")
    log(np.array2string(cluster_labels[:10]))
    log(f"Occupied SOM nodes: {len(unique_clusters)}")
    log(f"Empty SOM nodes: {GRID_ROWS * GRID_COLUMNS - len(unique_clusters)}")

    # 5. Evaluation metrics.
    score = None
    unique_label_count = len(np.unique(cluster_labels))

    if 2 <= unique_label_count <= len(X_scaled) - 1:
        score = float(
            silhouette_score(
                X_scaled,
                cluster_labels,
                metric="euclidean",
                sample_size=min(1000, len(X_scaled)),
                random_state=RANDOM_SEED,
            )
        )

    quantization_error = float(som.quantization_error(X_scaled))
    topographic_error = float(som.topographic_error(X_scaled))

    log("\n5. EVALUATION METRICS")
    log(
        "Silhouette score (sampled up to 1000 observations): "
        + (f"{score:.6f}" if score is not None else "Not defined")
    )
    log(f"Quantization error: {quantization_error:.6f}")
    log(f"Topographic error: {topographic_error:.6f}")

    if score is None:
        log(
            "Silhouette score could not be calculated because the number "
            "of distinct BMU labels is outside the valid range."
        )

    # 6. Save assignments.
    results = pd.DataFrame({
        "Actual Digit": y,
        "BMU X": bmu_coordinates[:, 0],
        "BMU Y": bmu_coordinates[:, 1],
        "Cluster Label": cluster_labels,
    })
    results.to_csv(OUTPUT_DIR / "som_cluster_assignments.csv", index=False)

    cluster_summary = (
        results.groupby("Cluster Label")
        .agg(
            Sample_Count=("Actual Digit", "size"),
            BMU_X=("BMU X", "first"),
            BMU_Y=("BMU Y", "first"),
            Most_Common_Digit=("Actual Digit", lambda values: int(values.mode().iloc[0])),
        )
        .reset_index()
    )
    cluster_summary.to_csv(OUTPUT_DIR / "som_cluster_summary.csv", index=False)

    metrics = pd.DataFrame([
        {"Metric": "Dataset samples", "Value": len(X)},
        {"Metric": "Input features", "Value": X.shape[1]},
        {"Metric": "SOM rows", "Value": GRID_ROWS},
        {"Metric": "SOM columns", "Value": GRID_COLUMNS},
        {"Metric": "Total SOM nodes", "Value": GRID_ROWS * GRID_COLUMNS},
        {"Metric": "Occupied SOM nodes", "Value": len(unique_clusters)},
        {"Metric": "Empty SOM nodes", "Value": GRID_ROWS * GRID_COLUMNS - len(unique_clusters)},
        {"Metric": "Training iterations", "Value": ITERATIONS},
        {"Metric": "Silhouette score", "Value": score if score is not None else "Not defined"},
        {"Metric": "Quantization error", "Value": quantization_error},
        {"Metric": "Topographic error", "Value": topographic_error},
    ])
    metrics.to_csv(OUTPUT_DIR / "som_metrics.csv", index=False)

    log("\n6. FIRST 15 CLUSTER ASSIGNMENTS")
    log(results.head(15).to_string(index=False))

    log("\n7. CLUSTER SIZES")
    log(cluster_summary[["Cluster Label", "Sample_Count"]].to_string(index=False))

    # 7. Plot SOM U-Matrix / distance map.
    plt.figure(figsize=(9, 8))
    distance_map = som.distance_map().T
    image = plt.pcolor(distance_map, cmap="bone_r", edgecolors="gray", linewidth=0.35)
    plt.colorbar(image, label="Normalized Mean Distance")
    plt.title("Self-Organizing Map Distance Map (U-Matrix)")
    plt.xlabel("SOM X-axis")
    plt.ylabel("SOM Y-axis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "som_distance_map.png", dpi=180)
    plt.close()

    # 8. Plot digit classes on the SOM grid.
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X"]
    plt.figure(figsize=(11, 9))

    for digit in range(10):
        mask = y == digit
        coords = bmu_coordinates[mask]
        plt.scatter(
            coords[:, 0] + 0.5,
            coords[:, 1] + 0.5,
            marker=markers[digit],
            s=34,
            facecolors="none",
            linewidths=1.0,
            label=f"Digit {digit}",
        )

    plt.xlim(0, GRID_ROWS)
    plt.ylim(0, GRID_COLUMNS)
    plt.xticks(np.arange(0, GRID_ROWS + 1))
    plt.yticks(np.arange(0, GRID_COLUMNS + 1))
    plt.grid(True, alpha=0.3)
    plt.title("Handwritten Digit Classes Mapped on the SOM Grid")
    plt.xlabel("SOM X-axis")
    plt.ylabel("SOM Y-axis")
    plt.legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "som_digit_class_map.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 9. Plot the number of observations mapped to each node.
    occupancy = np.zeros((GRID_ROWS, GRID_COLUMNS), dtype=int)
    for row, column in bmu_coordinates:
        occupancy[row, column] += 1

    plt.figure(figsize=(9, 8))
    image = plt.imshow(occupancy.T, origin="lower", cmap="viridis", aspect="equal")
    plt.colorbar(image, label="Number of Samples")
    plt.xticks(range(GRID_ROWS))
    plt.yticks(range(GRID_COLUMNS))
    plt.xlabel("SOM X-axis")
    plt.ylabel("SOM Y-axis")
    plt.title("SOM Node Occupancy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "som_node_occupancy.png", dpi=180)
    plt.close()

    # 10. Plot silhouette score when defined.
    plt.figure(figsize=(7, 5))
    if score is not None:
        plt.bar(["SOM BMU Clusters"], [score])
        plt.ylabel("Silhouette Score")
        plt.ylim(-1, 1)
        plt.title("Silhouette Score of SOM BMU Assignments")
        plt.text(0, score, f"{score:.3f}", ha="center",
                 va="bottom" if score >= 0 else "top")
    else:
        plt.text(
            0.5, 0.5,
            "Silhouette score is undefined for these cluster labels.",
            ha="center", va="center", wrap=True
        )
        plt.axis("off")
        plt.title("SOM Silhouette Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "som_silhouette_score.png", dpi=180)
    plt.close()

    # 11. Report summary.
    log("\n8. SUMMARY")
    log(f"Dataset samples: {len(X)}")
    log(f"Features per sample: {X.shape[1]}")
    log(f"SOM grid: {GRID_ROWS} x {GRID_COLUMNS}")
    log(f"Total nodes: {GRID_ROWS * GRID_COLUMNS}")
    log(f"Occupied nodes: {len(unique_clusters)}")
    log(f"Silhouette score: {score if score is not None else 'Not defined'}")
    log(f"Quantization error: {quantization_error:.6f}")
    log(f"Topographic error: {topographic_error:.6f}")

    report_path = OUTPUT_DIR / "experiment_output.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    log("\n9. GENERATED FILES")
    output_files = sorted(p for p in OUTPUT_DIR.iterdir() if p.is_file())
    for path in output_files:
        log(f"{path.name} ({path.stat().st_size:,} bytes)")

    # Rewrite the report after listing all generated files.
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    log("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
