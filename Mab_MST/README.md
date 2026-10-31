# Lab Report — SOM-Based Classification & Kernel Comparison

---

## 1. Aim
To implement a Self-Organizing Map (SOM) based classification model and investigate the impact of different SVM kernel functions on classification performance.

---

## 2. Objective
- Build a SOM to learn a topology-preserving representation of the input data.
- Use Best Matching Unit (BMU) weight vectors as transformed features.
- Train SVM classifiers with **linear, RBF, polynomial, and sigmoid** kernels on SOM features.
- Compare classification accuracy and identify the best-performing kernel.

---

## 3. Precise Description (max 5 lines)
A Self-Organizing Map is an unsupervised neural network that projects high-dimensional data onto a low-dimensional grid while preserving topological relationships. Each sample maps to its Best Matching Unit (BMU), whose weight vector serves as a compact, denoised feature. These SOM features are fed to an SVM classifier, and four kernel functions are evaluated. The experiment measures how kernel choice affects classification accuracy on SOM-projected data versus raw data. Results show non-linear kernels (RBF) generally outperform linear ones on SOM features.

---

## 4. Methodology
1. **Data Loading** — Load Iris dataset (150 samples, 4 features, 3 classes).
2. **Preprocessing** — StandardScaler normalization + 70/30 stratified train–test split.
3. **SOM Training** — 10×10 grid, Gaussian neighborhood, σ=1.2, η=0.5, 2000 iterations.
4. **Feature Transformation** — Replace each sample with its BMU weight vector.
5. **Classification** — Train 4 SVMs (linear, rbf, poly, sigmoid) on SOM features AND raw features.
6. **Evaluation** — Accuracy, confusion matrix, classification report.
7. **Visualization** — Bar chart, U-Matrix, confusion matrices.

---

## 5. Precise Code Steps

| Step | Code Action |
|------|-------------|
| 1 | `load_iris()` → split X, y |
| 2 | `StandardScaler().fit_transform(X)` |
| 3 | `train_test_split(test_size=0.30, stratify=y)` |
| 4 | `MiniSom(10, 10, 4, sigma=1.2, lr=0.5, 'gaussian')` |
| 5 | `som.random_weights_init(X_tr)` → `som.train_random(X_tr, 2000)` |
| 6 | For each x → `som.winner(x)` → fetch `W[r,c]` as new feature |
| 7 | Loop kernels `['linear','rbf','poly','sigmoid']` → `SVC(kernel=k).fit()` |
| 8 | `accuracy_score(y_te, y_pred)` for SOM & raw features |
| 9 | Plot bar chart, U-Matrix, 2×2 confusion matrices → save to `outputs/` |

---

## 6. Output (Drawn by Pencil)

**Fig. 1 — Kernel Comparison (Bar Chart)**

```
Accuracy
1.0 |  ██        ██   ██        ██
0.9 |  ██        ██   ██        ██
0.8 |  ██        ██   ██        ██
0.7 |  ██        ██   ██        ██
0.6 |  ██        ██   ██        ██
0.5 |  ██        ██   ██        ██
0.4 |  ██        ██   ██        ██
0.3 |  ██        ██   ██        ██   ██
0.2 |  ██        ██   ██        ██   ██
0.1 |  ██        ██   ██        ██   ██
    +------------------------------------
     linear     rbf    poly     sigmoid
      ▓ = SOM+SVM     ▒ = Raw SVM
```

**Fig. 2 — SOM U-Matrix (sketch)**

```
   y↑
   9 │ . . . . . . . . . .
   8 │ . . o . . . o . . .          o = Setosa
   7 │ . o o o . . . . s .          s = Versicolor
   6 │ . . o . . . . s s s          D = Virginica
   5 │ . . . . . . s s D D
   4 │ . . . . . s D D D .          Darker cell = larger
   3 │ . . . . s D D D . .          neuron distance
   2 │ . . . . . D D . . .
   1 │ . . . . . . . . . .
   0 │ . . . . . . . . . .
     +--------------------→ x
      0 1 2 3 4 5 6 7 8 9
```

**Fig. 3 — Confusion Matrices (2×2 grid)**

```
linear (0.911)     rbf (0.956)
[15  0  0]         [15  0  0]
[ 0 14  1]         [ 0 15  0]
[ 0  3 12]         [ 0  2 13]

poly (0.933)       sigmoid (0.311)
[15  0  0]         [15  0  0]
[ 0 14  1]         [ 0  0 15]
[ 0  2 13]         [ 0  0 15]
```

**Result Table**

| Kernel | SOM+SVM Acc | Raw SVM Acc |
|--------|-------------|-------------|
| linear | 0.911 | 0.956 |
| **rbf** | **0.956** | 0.956 |
| poly | 0.933 | 0.933 |
| sigmoid | 0.311 | 0.289 |

**SOM Errors:** Quantization = 0.0380, Topographic = 0.0286

---

## 7. Learning Outcomes

1. **Understood SOM mechanics** — competitive learning, BMU selection, Gaussian neighborhood, and topology preservation.
2. **Learned dimensionality reduction** — SOM acts as a non-linear quantizer producing compact prototype features.
3. **Kernel impact analysis** — RBF kernel is the most robust on SOM features; sigmoid fails due to non-PSD kernel issues.
4. **Feature engineering insight** — SOM features reduce noise but can lose discriminative detail, hence raw SVM matched SOM+RBF.
5. **Evaluation skills** — Practiced accuracy, confusion matrices, U-Matrix, quantization & topographic error.
6. **Practical toolkit** — Hands-on use of `minisom`, `sklearn.svm.SVC`, and Matplotlib visualization pipeline.