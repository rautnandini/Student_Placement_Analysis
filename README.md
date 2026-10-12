# Student Placement Clustering Analysis

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Verification-100%25%20Reproducible-brightgreen.svg)]()

A comprehensive data mining and unsupervised machine learning study analyzing student academic determinants and placement outcomes. This project evaluates **four clustering paradigms** across **four distinct feature selection and dimensionality reduction representations**, executing a matrix of **16 controlled experiments** evaluated against multi-metric internal validation indices.

---

## 📌 Executive Summary

While academic placement analysis is conventionally addressed using supervised classification, this project explores **unsupervised pattern discovery** to identify natural student academic-employability archetypes. 

### Key Empirical Findings:
- **Best Overall Clustering Algorithm**: **Agglomerative Hierarchical Clustering** (Overall Rank: 1st, Mean Silhouette: `0.1632`, Mean Davies-Bouldin: `1.9637`, Mean Runtime: `0.0022 s`).
- **Best Dimensionality Representation**: **PCA with 95% Explained Variance** achieved the highest clustering compactness and separation across all models (Mean Silhouette: `0.1726`, Davies-Bouldin: `2.0006`).
- **Optimal Centroid Configuration**: **Mutual Information Feature Selection + K-Means** attained the highest single Calinski-Harabasz score (`44.0954`), proving that non-linear feature ranking aligns effectively with centroid partitioning.

---

## 📁 Repository Structure

```
student-placement-clustering-analysis/
│
├── README.md                      # Comprehensive project documentation and benchmark guide
├── requirements.txt               # Pinned Python dependencies for exact reproducibility
├── .gitignore                     # Optimized gitignore for Python, Jupyter, and virtual environments
│
├── data/
│   ├── raw/
│   │   └── Placement_Data_Original.csv    # Untouched source dataset (215 rows × 15 columns)
│   └── processed/
│       └── Placement_Data_Cleaned.csv     # Cleaned, encoded, and scaled dataset (215 rows × 17 columns)
│
├── notebooks/
│   └── student_placement_analysis.ipynb   # Fully executed, end-to-end Jupyter Notebook
│
├── outputs/
│   ├── figures/                   # 44 high-resolution (300 DPI) publication-ready plots
│   │   ├── 01_outlier_boxplots.png
│   │   ├── 02_numeric_distributions.png
│   │   ├── 03_correlation_heatmap_encoded.png
│   │   ├── 04_numeric_pairplot.png
│   │   ├── 05_placement_status_distribution.png
│   │   ├── 06_feature_correlation_matrix.png
│   │   ├── 07_mutual_information_scores.png
│   │   ├── 08_pca_explained_variance.png
│   │   ├── 09_elbow_method_*.png
│   │   ├── 10_pca_scatter_*.png
│   │   ├── 11_clusters_*.png
│   │   ├── 12_heatmap_*.png
│   │   ├── 14_boxplot_*.png
│   │   ├── 15_elbow_curve_*.png
│   │   └── 16_silhouette_*.png
│   └── results/                   # Tabular results and clustered experiment datasets
│       ├── Algorithm_Comparison.csv
│       ├── Algorithm_Ranking.csv
│       ├── Dataset_Version_Comparison.csv
│       ├── Clustering_Evaluation_All_16_Experiments.csv
│       ├── Clustered_Original.csv
│       ├── Clustered_CorrelationSelection.csv
│       ├── Clustered_MutualInformation.csv
│       └── Clustered_PCA95.csv
│
└── docs/
    └── methodology.md             # In-depth technical methodology and mathematical definitions
```

---

## 📊 Dataset & Licensing Compliance

### Dataset Overview
- **Source**: Kaggle Campus Recruitment Dataset by Ben Roshan (`benroshan/factors-affecting-campus-placement`).
- **Instances**: 215 students.
- **Attributes**: 15 raw features including secondary education percentages (`ssc_p`), higher secondary percentages (`hsc_p`), degree percentages (`degree_p`), work experience (`workex`), employability test scores (`etest_p`), MBA percentage (`mba_p`), MBA specialization (`specialisation`), and placement status (`status`).

### Legal & Redistribution Assessment
- **License**: **CC0: Public Domain Dedication**.
- **Redistribution Rights**: Public domain dedication allows unrestricted distribution, modification, adaptation, and sharing. The raw untouched dataset is included in `data/raw/` in full compliance with open-source redistribution terms.

---

## ⚙️ Methodology Overview

The analytical pipeline is structured into 9 modular phases:

```
[Raw Data] ➔ [Data Preprocessing] ➔ [Feature Representations] ➔ [Clustering Suite] ➔ [Evaluation & Ranking]
```

1. **Phase 2 — Exploratory Study**: Auditing data distributions, schema integrity, and duplicate absence.
2. **Phase 3 — Data Cleaning & Preprocessing**:
   - Imputed missing salary values for unplaced candidates to `0` (logical imputation).
   - Flagged outliers using the Interquartile Range (IQR) method.
   - Applied `LabelEncoder` for binary attributes and `pd.get_dummies` for multi-class nominal features.
   - Normalized all candidate clustering predictors using `StandardScaler`.
3. **Phase 4 — Feature Selection & Dimensionality Reduction**:
   - **Version 1 (Original)**: Standardized 14-feature baseline.
   - **Version 2 (Correlation Selection)**: Pairwise collinearity filtering ($|r| < 0.85$, 13 features).
   - **Version 3 (Mutual Information)**: Top-8 features ranked by mutual information with placement status.
   - **Version 4 (PCA 95%)**: Principal Component Analysis reducing dimensions to 10 orthogonal components capturing $\ge 95\%$ variance.
4. **Phase 5 — Clustering Suite (16 Experiments)**:
   - **K-Means**: $K=3$, `init='k-means++'`, `n_init=10`.
   - **Agglomerative Hierarchical**: $K=3$, Ward linkage, Euclidean distance.
   - **DBSCAN**: Dynamic $\varepsilon$ estimated via 5-NN distance graph (90th percentile), `min_samples=5`.
   - **Gaussian Mixture Models (GMM)**: $K=3$, full covariance matrix.
5. **Phase 6 & 8 — Metric Evaluation & Multi-Criteria Ranking**:
   - Internal validation via **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Score**.
   - Computational cost measured via execution time.
6. **Phase 7 — Visual Diagnostics**: PCA 2D projections, correlation heatmaps, boxplots, elbow curves, and silhouette diagrams.

---

## 📈 Experimental Results

### Overall Algorithm Comparison

| Algorithm | Mean Silhouette Score ↑ | Mean Davies-Bouldin Index ↓ | Mean Calinski-Harabasz Score ↑ | Mean Execution Time | Overall Rank |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Agglomerative Hierarchical** | **0.1632** | **1.9637** | 31.9861 | **0.0022 s** | **1st (Rank 4.0)** |
| **K-Means** | 0.1436 | 2.0479 | **36.8311** | 0.0349 s | **2nd (Rank 5.0)** |
| **Gaussian Mixture (GMM)** | 0.1052 | 2.5729 | 19.6467 | 0.0158 s | **3rd (Rank 9.0)** |
| **DBSCAN** | N/A* | N/A* | N/A* | 0.0038 s | Unranked |

*\* Note: DBSCAN collapsed to a single continuous cluster under density estimation in this feature space.*

### Dataset Representation Comparison

| Dataset Representation | Mean Silhouette Score ↑ | Mean Davies-Bouldin Index ↓ | Mean Calinski-Harabasz Score ↑ | Mean Time |
|:---|:---:|:---:|:---:|:---:|
| **PCA 95% (10 PCs)** | **0.1726** | **2.0006** | **34.5012** | **0.0135 s** |
| **Original (14 Features)** | 0.1437 | 2.1813 | 26.4228 | 0.0137 s |
| **Mutual Information (Top 8)** | 0.1347 | 2.0499 | 33.8119 | 0.0137 s |
| **Correlation Selection (13 Features)** | 0.0985 | 2.5476 | 23.2159 | 0.0158 s |

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone or Download Repository
```bash
git clone https://github.com/<your-username>/student-placement-clustering-analysis.git
cd student-placement-clustering-analysis
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Analysis

#### Option A: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/student_placement_analysis.ipynb
```
*(Or open in VS Code / JupyterLab and click "Run All")*

#### Option B: Headless Command-Line Execution
To execute the notebook end-to-end and update all figures and CSV results in place:
```bash
python -m jupyter nbconvert --execute --inplace notebooks/student_placement_analysis.ipynb
```

---

## 🔒 Security, Audit & Integrity Verification

- **Data Privacy**: No private credentials, API keys, or personally identifiable data (PII) exist in the repository.
- **Path Resolution**: The notebook dynamically resolves root directories using `pathlib.Path`, operating seamlessly whether executed from the project root or from within the `notebooks/` folder.
- **Zero Temporary File Clutter**: Intermediate datasets are managed in memory during execution without polluting the root repository.
- **Preserved Originals**: Untouched raw data is safeguarded in `data/raw/` and never mutated by the pipeline.

---

## 📚 Technical Documentation

For complete mathematical formulations, hyperparameter justifications, and detailed academic context, refer to:
- [Technical Methodology Guide](docs/methodology.md)

---

## 📄 License & Attribution

- **Dataset**: [Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) by Ben Roshan (Licensed under **CC0: Public Domain**).
- **Code & Project Structure**: Open source and available under the [MIT License](LICENSE) (or CC0).
