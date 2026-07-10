# BIG-DATA-FINAL-PROYECT-2025-01
-3-
# Clinical Data Intelligence Pipeline — Healthcare Dataset

Big Data final project: an end-to-end pipeline over a synthetic healthcare dataset (Kaggle, 55,500 admission records), covering preprocessing, dimensionality reduction, patient clustering, a predictive ranking engine, and graph-based structural analysis, served through an interactive Streamlit dashboard.

**Course:** Big Data — Universidad Peruana de Ciencias Aplicadas
**Section:** 18519

## Project Overview

The pipeline answers one core question: *is it possible to cluster patient profiles and treatment pathways to identify patterns between medical conditions, medication types, and hospital admission outcomes?*

It is built in five sequential stages:

| Stage | What it does | Key output |
|---|---|---|
| 1. Preprocessing | Loads raw data, deduplicates, engineers features (`Cost_Per_Day`, label decoding) | `data/interim/healtcare_processedv2.csv` |
| 2. PCA / Clustering | Reduces a 33-feature patient matrix to 14–15 principal components; K-Means and DBSCAN segment patients | `data/processed/healthcare_pca_clusters.csv`, `artifacts/models/` |
| 3. Ranking Engine | Ensemble of Random Forest + CatBoost scores admissions by predicted probability of an Abnormal test result | ranked priority queue (CSV) |
| 4. Graph Analytics | Bipartite condition↔medication graph, edges filtered by statistical significance (lift ≥ 1.05) | `artifacts/graph/clinical_graph.gexf` |
| 5. Dashboard | Streamlit app exposing EDA, PCA, Clustering, Priorization, and the interactive Clinical Graph | live app (see link below) |

## Repository Structure

```
.
├── data/
│   ├── raw/               # Original, unmodified source data (healthcare_dataset.csv)
│   ├── interim/           # Cleaned/preprocessed data (post-dedup, feature engineering)
│   └── processed/         # Final modeling-ready data (PCA clusters, etc.)
├── notebooks/             # Analysis notebooks, one per pipeline stage
├── src/                   # Reusable Python modules (data_utils, graph_utils, clustering_utils, etc.)
├── artifacts/
│   ├── models/            # Trained models (K-Means)
│   └── graph/             # Exported graph files (clinical_graph.gexf)
├── reports/               # Final technical report and supporting documents
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/SofiaGMiranda/BIG-DATA-FINAL-PROYECT-2025-01.git
cd BIG-DATA-FINAL-PROYECT-2025-01

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Reproducing the Pipeline

See [`RUNBOOK.md`](./RUNBOOK.md) for the full, step-by-step, copy-pasteable sequence of commands to regenerate every artifact in this repository from the raw CSV. In short, notebooks must be run in this order:

1. `notebooks/01_preprocessing.ipynb` — cleaning, feature engineering
2. `notebooks/02_pca_clustering.ipynb` — PCA, K-Means, DBSCAN
3. `notebooks/03_ranking_engine.ipynb` — baselines, ensemble ranking model
4. `notebooks/04_graph_analytics.ipynb` — lift-filtered clinical graph

> Update the file names above to match your actual `notebooks/` folder if they differ.

## Interactive Dashboard

The clinical graph and other analysis views are available as a live Streamlit app:

🔗 **[Live Dashboard](https://big-data-final-proyect-2025-01-n4jxjkldy3up8a6wzmkpyk.streamlit.app/)**

To run it locally instead:

```bash
cd src
streamlit run app.py
```

> **Note:** `app.py` currently relies on relative paths (`../data/...`) that resolve based on the working directory, not the file location — so it must be launched from inside `src/`. See `graph_utils.py` for the `Path(__file__)`-based pattern used to make path resolution independent of the working directory; applying the same pattern to the other modules is tracked as an open item in Limitations and Future Work.

## Key Findings (Summary)

- **Clustering:** K-Means (K=3–4) identified operationally distinct patient segments; validated with silhouette score and the Calinski-Harabasz index.
- **Ranking Engine:** The Random Forest + CatBoost ensemble reached Precision@10 = 0.600, tripling the random-ranking baseline (0.200).
- **Graph Analytics:** Filtering the condition–medication graph by statistical significance (lift ≥ 1.05) reduced density from 0.55 to 0.20, and produced a PageRank ranking meaningfully different from raw popularity (Spearman ρ = 0.38 vs. 0.97 on the unfiltered graph).

Full methodology, evaluation protocol, limitations, and the monitoring/operationalization plan are documented in [`reports/`](./reports).

## Team

- Gomez Rubina, Luis David (U20221C621)
- Miranda Cardenas, Sofia Gabriel (U20191C439)
- Olivera Alvarez, Lizbeth Teresita (U201616851)

## License / Data Source

Dataset: [Healthcare Dataset (Kaggle, synthetic)](https://www.kaggle.com/datasets/prasad22/healthcare-dataset). Used under academic license for coursework; contains no real patient data (see Ethics and Access Note in the final report).
