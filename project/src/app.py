import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from data_utils import process_data
from charts import plot_categorical, plot_numerical
from modeling_utils import run_care_prioritization_section
from graph_utils import run_graph_section

from pca_utils import (
    build_patient_feature_matrix,
    run_pca,
    plot_variance_explained,
    show_components_table,
    plot_pca_scatter,
    show_full_matrix,
)

from clustering_utils import (
    kmeans_clustering_section,
    dbscan_clustering_section,
    show_validation_table,
    show_failure_analysis,
)

st.set_page_config(page_title="Clinical analysis", layout="wide", page_icon="💉​")

st.title("🏥 Clinical Analytics System — Patient Prioritization & Segmentation")
st.caption("Healthcare data pipeline: data quality, PCA, clustering, clinical prioritization model, and condition-medication association graph")


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "healthcare_dataset.csv"

#uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

NEW_FEATURES = ["Length of Stay", "Admission Day of Week", "Admission Month", "Admission Year"]

def contar_outliers_iqr(serie):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((serie < lim_inf) | (serie > lim_sup)).sum()

#if uploaded_file:
#raw_len = pd.read_csv(uploaded_file).shape[0]
#uploaded_file.seek(0)

#df = process_data(uploaded_file)

raw_len = pd.read_csv(RAW_DATA).shape[0]

df = process_data(RAW_DATA)

cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_plot = [c for c in cat_cols if c not in ["Name", "Doctor", "Hospital"]]

total_nulos = df.isnull().sum().sum()
total_duplicados = raw_len - len(df)
total_outliers = sum(contar_outliers_iqr(df[c]) for c in num_cols)

    # ---- Manual navigation ----
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "🔍 Diagnostics"

st.session_state.active_tab = st.radio(
        "Navigation",
        ["🔍 Diagnostics", "📈 EDA", "🧬 PCA", "🧩 Clustering", "🎯 Prioritization","🕸️ Graph"],
        horizontal=True,
        label_visibility="collapsed",
        index=["🔍 Diagnostics", "📈 EDA", "🧬 PCA", "🧩 Clustering", "🎯 Prioritization","🕸️ Graph"].index(st.session_state.active_tab),
    )

st.divider()

if st.session_state.active_tab == "🔍 Diagnostics":
    st.subheader("Dataset Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Duplicates Removed", f"{total_duplicados:,}")
    c3.metric("Null Values", f"{total_nulos:,}")
    c4.metric("Outliers (IQR)", f"{total_outliers:,}")
    c5.metric("Total Columns", f"{df.shape[1]}")

    c6, c7 = st.columns(2)
    c6.metric("Categorical Variables", len(cat_cols))
    c7.metric("Numerical Variables", len(num_cols))

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("##### Preview (first 10 rows)")
        st.dataframe(df.head(10), width='stretch')

    with col_right:
        st.markdown("##### Nulls per Column")
        nulos_col = df.isnull().sum()
        nulos_col = nulos_col[nulos_col > 0].sort_values(ascending=False)
        if len(nulos_col):
            st.dataframe(nulos_col.rename("Nulls"), width='stretch')
        else:
            st.success("No null values ✅")

        st.divider()

        st.markdown("##### Newly Generated Variables")
        st.caption("Automatically created during dataset processing")
        cols_new = st.columns(len(NEW_FEATURES))
        for i, feat in enumerate(NEW_FEATURES):
            with cols_new[i]:
                with st.container(border=True):
                    st.markdown(f"**{feat}**")
                    st.caption(str(df[feat].dtype))

        with st.expander("View sample of the new variables"):
            st.dataframe(df[NEW_FEATURES].head(10), width='stretch')

elif st.session_state.active_tab == "📈 EDA":
    st.subheader("Variable Exploration")

    st.markdown("### Categorical Variable")
    sel_cat = st.selectbox("Category:", cat_cols_plot)
    plot_categorical(df, sel_cat)

    st.divider()

    st.markdown("### Numerical Variable")
    sel_num = st.selectbox("Numeric:", num_cols)
    plot_numerical(df, sel_num)

elif st.session_state.active_tab == "🧬 PCA":
    st.subheader("Dimensionality Reduction — PCA")

    with st.spinner("Building per-patient feature matrix..."):
        patient_feature_matrix = build_patient_feature_matrix(df)
        pca_result = run_pca(patient_feature_matrix)

    st.markdown("### Explained Variance")
    plot_variance_explained(pca_result)

    st.divider()

    show_components_table(pca_result)

    st.divider()

    st.markdown("### Patient Projection (PC1 vs PC2)")
    plot_pca_scatter(pca_result, patient_feature_matrix)

    st.divider()

    show_full_matrix(patient_feature_matrix)
    
elif st.session_state.active_tab == "🧩 Clustering":
    st.subheader("Patient Segmentation — Clustering")

    with st.spinner("Building feature matrix and PCA..."):
        patient_feature_matrix = build_patient_feature_matrix(df)
        pca_result = run_pca(patient_feature_matrix)

    st.markdown("### K-Means")
    kmeans_labels, kmeans_model = kmeans_clustering_section(pca_result, patient_feature_matrix)

    st.divider()

    st.markdown("### DBSCAN")
    dbscan_labels, dbscan_model = dbscan_clustering_section(pca_result, patient_feature_matrix)

    st.divider()

    st.markdown("### Validation — K-Means vs DBSCAN")
    show_validation_table(pca_result, kmeans_labels, dbscan_labels)

    st.divider()

    st.markdown("### Failure and Anomaly Analysis")
    show_failure_analysis(pca_result, patient_feature_matrix, kmeans_labels, dbscan_labels)
    
elif st.session_state.active_tab == "🎯 Prioritization":
    st.subheader("Clinical Prioritization Engine")
    run_care_prioritization_section()

elif st.session_state.active_tab == "🕸️ Graph":
    st.subheader("Relationship Network Analysis")
    run_graph_section()

#else:
    #st.info("⬅️ Upload a CSV file from the sidebar to start the audit.")