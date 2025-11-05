# 🧩 Cluster Analysis with PCA Visualization

A machine learning project that performs **unsupervised clustering** on datasets and visualizes the results using **Principal Component Analysis (PCA)**.  
This project explores how different clustering algorithms group similar data points and uses PCA to reduce high-dimensional data into **2D and 3D visualizations** for clear pattern analysis.

---

## 🚀 Features

- 🧠 Implements **K-Means**, **Hierarchical Clustering**, and **DBSCAN** algorithms  
- 📉 Reduces dimensionality using **PCA** for 2D and 3D visualization  
- 🎨 Generates interactive scatter plots with clear cluster boundaries  
- ⚙️ Includes **data preprocessing**, scaling, and normalization  
- 🔬 Compares clustering results with **inertia** and **silhouette scores**

---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Language** | Python |
| **Libraries** | Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| **Algorithms** | K-Means, Hierarchical Clustering, DBSCAN |
| **Visualization** | PCA (Principal Component Analysis) |

---

## 🧠 Objective

To apply **unsupervised learning** techniques to identify natural groupings within data and visualize them effectively using **PCA projections**.  
This helps in understanding high-dimensional datasets and revealing hidden patterns.



🧩 Workflow Overview

Data Preprocessing:

Load dataset

Handle missing values

Normalize or scale features

Clustering:

Apply K-Means, Hierarchical, and DBSCAN algorithms

Determine optimal number of clusters

PCA Transformation:

Reduce dimensions from high-D space to 2D/3D

Visualize clusters using scatter plots

Evaluation:

Compare clustering quality using inertia or silhouette score

├── KMEANS.ipynb                 # Jupyter Notebook (main implementation)
├── cluster_analysis.py          # Python script version
├── data.csv                     # Sample dataset (optional)
├── requirements.txt             # Dependencies
├── .gitignore                   # Ignored files (checkpoints, env, etc.)
└── README.md                    # Project documentation


Learning Outcomes

Implemented unsupervised machine learning algorithms

Gained experience with dimensionality reduction (PCA)

Learned to visualize complex data distributions

Compared clustering methods to evaluate performance
