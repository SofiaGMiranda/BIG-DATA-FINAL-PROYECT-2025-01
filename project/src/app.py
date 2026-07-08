import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="Healthcare Audit", layout="wide", page_icon="📊")

st.title("Auditoría y Exploración de Datos")
st.caption("Dashboard de calidad de datos y análisis exploratorio — dataset de salud")

uploaded_file = st.sidebar.file_uploader("Subir CSV", type=["csv"])

NEW_FEATURES = ["Length of Stay", "Admission Day of Week", "Admission Month", "Admission Year"]

def contar_outliers_iqr(serie):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((serie < lim_inf) | (serie > lim_sup)).sum()

if uploaded_file:
    raw_len = pd.read_csv(uploaded_file).shape[0]
    uploaded_file.seek(0)

    df = process_data(uploaded_file)

    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols_plot = [c for c in cat_cols if c not in ["Name", "Doctor", "Hospital"]]

    total_nulos = df.isnull().sum().sum()
    total_duplicados = raw_len - len(df)
    total_outliers = sum(contar_outliers_iqr(df[c]) for c in num_cols)

    # ---- Navegación manual ----
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "🔍 Diagnóstico"

    st.session_state.active_tab = st.radio(
        "Navegación",
        ["🔍 Diagnóstico", "📈 EDA", "🧬 PCA", "🧩 Clustering", "🎯 Priorización","🕸️ Grafo"],
        horizontal=True,
        label_visibility="collapsed",
        index=["🔍 Diagnóstico", "📈 EDA", "🧬 PCA", "🧩 Clustering", "🎯 Priorización","🕸️ Grafo"].index(st.session_state.active_tab),
    )

    st.divider()

    if st.session_state.active_tab == "🔍 Diagnóstico":
        st.subheader("Resumen del dataset")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Filas totales", f"{len(df):,}")
        c2.metric("Duplicados eliminados", f"{total_duplicados:,}")
        c3.metric("Valores nulos", f"{total_nulos:,}")
        c4.metric("Outliers (IQR)", f"{total_outliers:,}")
        c5.metric("Columnas totales", f"{df.shape[1]}")

        c6, c7 = st.columns(2)
        c6.metric("Variables categóricas", len(cat_cols))
        c7.metric("Variables numéricas", len(num_cols))

        st.divider()

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("##### Vista previa (primeras 10 filas)")
            st.dataframe(df.head(10), width='stretch')

        with col_right:
            st.markdown("##### Nulos por columna")
            nulos_col = df.isnull().sum()
            nulos_col = nulos_col[nulos_col > 0].sort_values(ascending=False)
            if len(nulos_col):
                st.dataframe(nulos_col.rename("Nulos"), width='stretch')
            else:
                st.success("Sin valores nulos ✅")

        st.divider()

        st.markdown("##### Nuevas variables generadas")
        st.caption("Creadas automáticamente durante el procesamiento del dataset")
        cols_new = st.columns(len(NEW_FEATURES))
        for i, feat in enumerate(NEW_FEATURES):
            with cols_new[i]:
                with st.container(border=True):
                    st.markdown(f"**{feat}**")
                    st.caption(str(df[feat].dtype))

        with st.expander("Ver muestra de las nuevas variables"):
            st.dataframe(df[NEW_FEATURES].head(10), width='stretch')

    elif st.session_state.active_tab == "📈 EDA":
        st.subheader("Exploración de variables")

        st.markdown("### Variable categórica")
        sel_cat = st.selectbox("Categoría:", cat_cols_plot)
        plot_categorical(df, sel_cat)

        st.divider()

        st.markdown("### Variable numérica")
        sel_num = st.selectbox("Numérico:", num_cols)
        plot_numerical(df, sel_num)

    elif st.session_state.active_tab == "🧬 PCA":
        st.subheader("Reducción de dimensionalidad — PCA")

        with st.spinner("Construyendo matriz de características por paciente..."):
            patient_feature_matrix = build_patient_feature_matrix(df)
            pca_result = run_pca(patient_feature_matrix)

        st.markdown("### Varianza explicada")
        plot_variance_explained(pca_result)

        st.divider()

        show_components_table(pca_result)

        st.divider()

        st.markdown("### Proyección de pacientes (PC1 vs PC2)")
        plot_pca_scatter(pca_result, patient_feature_matrix)

        st.divider()

        show_full_matrix(patient_feature_matrix)
    
    elif st.session_state.active_tab == "🧩 Clustering":
        st.subheader("Segmentación de pacientes — Clustering")

        with st.spinner("Construyendo matriz de características y PCA..."):
            patient_feature_matrix = build_patient_feature_matrix(df)
            pca_result = run_pca(patient_feature_matrix)

        st.markdown("### K-Means")
        kmeans_labels, kmeans_model = kmeans_clustering_section(pca_result, patient_feature_matrix)

        st.divider()

        st.markdown("### DBSCAN")
        dbscan_labels, dbscan_model = dbscan_clustering_section(pca_result, patient_feature_matrix)

        st.divider()

        st.markdown("### Validación — K-Means vs DBSCAN")
        show_validation_table(pca_result, kmeans_labels, dbscan_labels)

        st.divider()

        st.markdown("### Análisis de fallas y anomalías")
        show_failure_analysis(pca_result, patient_feature_matrix, kmeans_labels, dbscan_labels)
    
    elif st.session_state.active_tab == "🎯 Priorización":
        st.subheader("Motor de Priorización Clínica")
        run_care_prioritization_section()

    elif st.session_state.active_tab == "🕸️ Grafo":
        st.subheader("Análisis de Redes de Relaciones")
        run_graph_section()

else:
    st.info("⬅️ Sube un archivo CSV desde la barra lateral para comenzar la auditoría.")