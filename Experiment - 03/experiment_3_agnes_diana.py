from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS = 3


def diana_clustering(X, n_clusters=3):
    """
    A simple divisive hierarchical clustering implementation.

    Starts with all samples in one cluster and repeatedly splits
    the cluster with the largest diameter. Within the selected
    cluster, the most dissimilar point starts a splinter group.
    Remaining points move to that group when they are, on average,
    closer to the splinter group than to the remaining group.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X must contain at least two samples.")
    if not 1 < n_clusters <= len(X):
        raise ValueError("n_clusters must be between 2 and the sample count.")

    clusters = [list(range(len(X)))]

    while len(clusters) < n_clusters:
        # Choose the non-singleton cluster with the largest diameter.
        cluster_to_split_index = None
        max_diameter = -1.0

        for index, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue

            distances = pdist(X[cluster], metric="euclidean")
            diameter = float(np.max(distances)) if len(distances) else 0.0

            if diameter > max_diameter:
                max_diameter = diameter
                cluster_to_split_index = index

        if cluster_to_split_index is None:
            raise RuntimeError("DIANA could not find a cluster to split.")

        cluster_to_split = clusters.pop(cluster_to_split_index)
        sub_X = X[cluster_to_split]
        distance_matrix = squareform(pdist(sub_X, metric="euclidean"))

        # Start the splinter group with the most dissimilar point.
        average_distances = distance_matrix.mean(axis=1)
        splinter_local_index = int(np.argmax(average_distances))
        splinter_group = [cluster_to_split[splinter_local_index]]
        remaining_group = [
            point for i, point in enumerate(cluster_to_split)
            if i != splinter_local_index
        ]

        # Repeatedly move points closer to the splinter group.
        changed = True
        while changed and len(remaining_group) > 1:
            changed = False

            for point in remaining_group.copy():
                splinter_distance = float(np.mean([
                    np.linalg.norm(X[point] - X[other])
                    for other in splinter_group
                ]))

                other_points = [
                    other for other in remaining_group if other != point
                ]

                # A point with no other remaining neighbours stays put.
                if not other_points:
                    continue

                remaining_distance = float(np.mean([
                    np.linalg.norm(X[point] - other)
                    for other in other_points
                ]))

                if splinter_distance < remaining_distance:
                    splinter_group.append(point)
                    remaining_group.remove(point)
                    changed = True

        if not remaining_group or not splinter_group:
            raise RuntimeError("DIANA produced an empty cluster.")

        clusters.extend([remaining_group, splinter_group])

    labels = np.empty(len(X), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        labels[cluster] = cluster_id

    return labels


def save_dendrogram(linkage_matrix, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        show_contracted=True,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title("AGNES Dendrogram (Ward Linkage)")
    ax.set_xlabel("Samples or Merged Clusters")
    ax.set_ylabel("Euclidean Distance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_cluster_plot(X, labels, title, output_path, feature_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        s=55,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )
    ax.set_title(title)
    ax.set_xlabel(f"{feature_names[0]} (standardized)")
    ax.set_ylabel(f"{feature_names[1]} (standardized)")
    ax.grid(alpha=0.2)
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
    )
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 72)
    print("EXPERIMENT 3: AGNES AND DIANA HIERARCHICAL CLUSTERING")
    print("=" * 72)

    # 1. Load Iris dataset. Target labels are not used for clustering.
    iris = load_iris()
    X = iris.data
    feature_names = list(iris.feature_names)

    raw_df = pd.DataFrame(X, columns=feature_names)
    print("\n1. DATASET INFORMATION")
    print(f"Dataset: Iris")
    print(f"Samples: {X.shape[0]}")
    print(f"Numerical features: {X.shape[1]}")
    print("Features:", ", ".join(feature_names))
    print("\nFirst five rows:")
    print(raw_df.head().to_string(index=False))

    # 2. Scale features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    scaled_df.to_csv(OUTPUT_DIR / "iris_scaled_features.csv", index=False)

    # 3. AGNES dendrogram and clustering.
    print("\n2. AGNES: AGGLOMERATIVE NESTING")
    linkage_matrix = linkage(X_scaled, method="ward", metric="euclidean")
    save_dendrogram(
        linkage_matrix,
        OUTPUT_DIR / "agnes_dendrogram.png",
    )

    agnes = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        linkage="ward",
    )
    agnes_labels = agnes.fit_predict(X_scaled)
    agnes_score = silhouette_score(X_scaled, agnes_labels)

    print(f"Linkage method: Ward")
    print(f"Requested clusters: {N_CLUSTERS}")
    print(f"Actual clusters: {len(np.unique(agnes_labels))}")
    print(f"Silhouette score: {agnes_score:.4f}")
    print("Cluster sizes:", dict(zip(
        *np.unique(agnes_labels, return_counts=True)
    )))

    # 4. DIANA custom implementation.
    print("\n3. DIANA: DIVISIVE ANALYSIS")
    diana_labels = diana_clustering(X_scaled, n_clusters=N_CLUSTERS)
    diana_score = silhouette_score(X_scaled, diana_labels)

    print("Split strategy: largest-diameter cluster")
    print(f"Requested clusters: {N_CLUSTERS}")
    print(f"Actual clusters: {len(np.unique(diana_labels))}")
    print(f"Silhouette score: {diana_score:.4f}")
    print("Cluster sizes:", dict(zip(
        *np.unique(diana_labels, return_counts=True)
    )))

    # 5. Save labels and comparison results.
    results_df = raw_df.copy()
    results_df["AGNES Cluster"] = agnes_labels
    results_df["DIANA Cluster"] = diana_labels
    results_df.to_csv(OUTPUT_DIR / "iris_cluster_assignments.csv", index=False)

    comparison = pd.DataFrame([
        {
            "Method": "AGNES",
            "Approach": "Bottom-up",
            "Linkage_or_Split": "Ward linkage",
            "Number of Clusters": len(np.unique(agnes_labels)),
            "Silhouette Score": agnes_score,
        },
        {
            "Method": "DIANA",
            "Approach": "Top-down",
            "Linkage_or_Split": "Largest-diameter split",
            "Number of Clusters": len(np.unique(diana_labels)),
            "Silhouette Score": diana_score,
        },
    ])
    comparison.to_csv(OUTPUT_DIR / "clustering_comparison.csv", index=False)

    print("\n4. COMPARISON TABLE")
    print(comparison.to_string(index=False, formatters={
        "Silhouette Score": "{:.4f}".format
    }))

    # 6. Visualize clusters using the first two standardized features.
    save_cluster_plot(
        X_scaled,
        agnes_labels,
        "AGNES Clustering on Iris (First Two Features)",
        OUTPUT_DIR / "agnes_clusters.png",
        feature_names,
    )
    save_cluster_plot(
        X_scaled,
        diana_labels,
        "DIANA Clustering on Iris (First Two Features)",
        OUTPUT_DIR / "diana_clusters.png",
        feature_names,
    )

    # 7. Compare silhouette scores graphically.
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        comparison["Method"],
        comparison["Silhouette Score"],
        width=0.55,
    )
    ax.set_title("Silhouette Score: AGNES vs DIANA")
    ax.set_ylabel("Silhouette Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "silhouette_score_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 8. Write a detailed text report.
    report_lines = [
        "=" * 72,
        "EXPERIMENT 3: AGNES AND DIANA HIERARCHICAL CLUSTERING",
        "=" * 72,
        "",
        "AIM",
        "Implement AGNES and DIANA for clustering the Iris dataset.",
        "",
        "DATASET",
        "Iris dataset from scikit-learn.",
        f"Samples: {X.shape[0]}",
        f"Features: {X.shape[1]}",
        "Ground-truth species labels were not used to fit the clusters.",
        "Features were standardized using StandardScaler.",
        "",
        "AGNES",
        "Approach: bottom-up agglomerative clustering.",
        "Linkage: Ward; distance metric: Euclidean.",
        f"Number of clusters: {len(np.unique(agnes_labels))}",
        f"Silhouette score: {agnes_score:.4f}",
        f"Cluster sizes: {dict(zip(*np.unique(agnes_labels, return_counts=True)))}",
        "",
        "DIANA",
        "Approach: top-down divisive clustering.",
        "Split rule: split the cluster with the largest diameter.",
        "Splinter rule: move points closer on average to the splinter group.",
        f"Number of clusters: {len(np.unique(diana_labels))}",
        f"Silhouette score: {diana_score:.4f}",
        f"Cluster sizes: {dict(zip(*np.unique(diana_labels, return_counts=True)))}",
        "",
        "COMPARISON",
        comparison.to_string(index=False, formatters={
            "Silhouette Score": "{:.4f}".format
        }),
        "",
        "INTERPRETATION",
        "Silhouette scores range from -1 to 1; higher values generally indicate",
        "better separation and cohesion for the produced clusters.",
        "Cluster IDs are arbitrary and do not correspond to Iris species names.",
        "The custom DIANA routine is a simplified divisive implementation, not",
        "a claim of exact equivalence to every published DIANA variant.",
        "",
        "GENERATED OUTPUTS",
        "experiment_output.txt",
        "iris_scaled_features.csv",
        "iris_cluster_assignments.csv",
        "clustering_comparison.csv",
        "agnes_dendrogram.png",
        "agnes_clusters.png",
        "diana_clusters.png",
        "silhouette_score_comparison.png",
    ]

    report_path = OUTPUT_DIR / "experiment_output.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n5. OUTPUT FILES")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"{path.name:38s} {path.stat().st_size:>9,} bytes")

    print(f"\nReport saved to: {report_path}")
    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
