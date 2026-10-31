"""
SOM-based Classification with Kernel Function Comparison
Author: AML Lab
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # safe for headless terminal
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("outputs", exist_ok=True)
np.random.seed(42)

# ------------------------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# ------------------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")

# ------------------------------------------------------------------
# 2. TRAIN SELF-ORGANIZING MAP
# ------------------------------------------------------------------
GRID = (10, 10)
som = MiniSom(
    x=GRID[0], y=GRID[1], input_len=X.shape[1],
    sigma=1.2, learning_rate=0.5, neighborhood_function='gaussian',
    random_seed=42
)
som.random_weights_init(X_tr)
print("Training SOM ...")
som.train_random(X_tr, num_iteration=2000, verbose=False)
print("SOM training complete.")

# ------------------------------------------------------------------
# 3. SOM-BASED FEATURE TRANSFORMATION (BMU weight vectors)
# ------------------------------------------------------------------
W = som.get_weights()           # shape (grid_x, grid_y, n_features)

def som_transform(X):
    """Map each sample to its BMU's weight vector."""
    F = np.zeros((len(X), W.shape[2]))
    for i, x in enumerate(X):
        r, c = som.winner(x)
        F[i] = W[r, c]
    return F

X_tr_som = som_transform(X_tr)
X_te_som = som_transform(X_te)

# ------------------------------------------------------------------
# 4. KERNEL COMPARISON (on SOM features vs Raw features)
# ------------------------------------------------------------------
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
som_acc, raw_acc, som_pred = {}, {}, {}

print("\n" + "=" * 60)
print(f"{'Kernel':<10}{'SOM + SVM':<18}{'Raw SVM':<15}")
print("=" * 60)

for k in kernels:
    # --- on SOM features ---
    clf = SVC(kernel=k, C=1.0, gamma='scale', degree=3, random_state=42)
    clf.fit(X_tr_som, y_tr)
    p = clf.predict(X_te_som)
    som_acc[k]  = accuracy_score(y_te, p)
    som_pred[k] = p

    # --- baseline on raw features ---
    clf_raw = SVC(kernel=k, C=1.0, gamma='scale', degree=3, random_state=42)
    clf_raw.fit(X_tr, y_tr)
    raw_acc[k] = accuracy_score(y_te, clf_raw.predict(X_te))

    print(f"{k:<10}{som_acc[k]:<18.4f}{raw_acc[k]:<15.4f}")
print("=" * 60)

best_k = max(som_acc, key=som_acc.get)
print(f"\n>>> Best kernel on SOM features: {best_k}  (acc={som_acc[best_k]:.4f})")
print("\nClassification report for best kernel:\n")
print(classification_report(y_te, som_pred[best_k], target_names=iris.target_names))

# ------------------------------------------------------------------
# 5. PLOTS
# ------------------------------------------------------------------
# 5a. Bar comparison
fig, ax = plt.subplots(figsize=(9, 5))
xx = np.arange(len(kernels)); w = 0.35
b1 = ax.bar(xx - w/2, [som_acc[k] for k in kernels], w,
            label='SOM + SVM', color='#2E86AB')
b2 = ax.bar(xx + w/2, [raw_acc[k] for k in kernels], w,
            label='Raw SVM', color='#E63946')
ax.set_xticks(xx); ax.set_xticklabels(kernels)
ax.set_ylabel('Test Accuracy'); ax.set_ylim(0, 1.12)
ax.set_title('Impact of Kernel Function on Classification Accuracy')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015,
                f'{b.get_height():.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/kernel_comparison.png', dpi=150)
plt.close()

# 5b. Confusion matrices (one subplot per kernel)
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, k in zip(axes.ravel(), kernels):
    cm = confusion_matrix(y_te, som_pred[k])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f"Kernel = {k}   (acc = {som_acc[k]:.3f})", fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(iris.target_names, rotation=25)
    ax.set_yticklabels(iris.target_names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=12)
plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=150)
plt.close()

# 5c. SOM U-Matrix with class overlays
fig, ax = plt.subplots(figsize=(8, 8))
um = som.distance_map().T
pcm = ax.pcolor(um, cmap='bone_r')
plt.colorbar(pcm, ax=ax, label='Neuron distance')
markers = ['o', 's', 'D']
colors  = ['red', 'green', 'blue']
for i, x in enumerate(X):
    r, c = som.winner(x)
    ax.plot(r + 0.5, c + 0.5, markers[y[i]],
            markerfacecolor='none',
            markeredgecolor=colors[y[i]],
            markersize=10, markeredgewidth=1.8)
from matplotlib.lines import Line2D
legend = [Line2D([0],[0], marker=markers[i], color='w',
                 markerfacecolor='none', markeredgecolor=colors[i],
                 markersize=10, label=iris.target_names[i])
          for i in range(3)]
ax.legend(handles=legend, loc='upper right', framealpha=0.9)
ax.set_title('SOM U-Matrix with Class Labels')
ax.set_xlabel('SOM x'); ax.set_ylabel('SOM y')
plt.tight_layout()
plt.savefig('outputs/som_umatrix.png', dpi=150)
plt.close()

# 5d. Learning curve (quantization error)
q_err = som.quantization_error(X_tr)
t_err = som.topographic_error(X_tr)
print(f"\nSOM Quantization Error : {q_err:.4f}")
print(f"SOM Topographic Error  : {t_err:.4f}")
print("\nSaved plots -> outputs/kernel_comparison.png, "
      "confusion_matrices.png, som_umatrix.png")
