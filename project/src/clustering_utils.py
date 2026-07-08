import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score

plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'axes.edgecolor': '#888888',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'grid.color': '#444444',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# =======================================================================
# K-MEANS
# =======================================================================
@st.cache_data(show_spinner=False)
def kmeans_parameter_sweep(pca_array, k_min=2, k_max=10):
    """Barrido de K: inertia y silhouette score para cada K."""
    ks = range(k_min, k_max + 1)
    inertia_scores, silhouette_scores = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pca_array)
        inertia_scores.append(km.inertia_)
        silhouette_scores.append(silhouette_score(pca_array, labels))
    return pd.DataFrame({"k": list(ks), "inertia": inertia_scores, "silhouette": silhouette_scores})


def _plot_kmeans_sweep(results):
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(results["k"], results["inertia"], marker="o", linestyle="-", color="#5da5da")
        ax.set_title("Elbow Method (Inertia)", fontsize=12)
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        ax.set_xticks(results["k"])
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(results["k"], results["silhouette"], marker="o", linestyle="-", color="#f17c67")
        ax.set_title("Silhouette Score por K", fontsize=12)
        ax.set_xlabel("K")
        ax.set_ylabel("Silhouette")
        ax.set_xticks(results["k"])
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)


def _plot_kmeans_scatter(pca_array, labels, model):
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(pca_array[:, 0], pca_array[:, 1], c=labels, cmap='viridis',
                          s=40, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
               marker='X', s=250, c='red', edgecolor='black', linewidth=1.5, label='Centroides')
    ax.set_title("Clusters K-Means en espacio PCA (PC_1 vs PC_2)", fontsize=13)
    ax.set_xlabel("PC_1")
    ax.set_ylabel("PC_2")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def _plot_cluster_sizes(labels, title="Tamaño de clusters"):
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color='#5da5da', edgecolor='black')
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cantidad de observaciones")
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def kmeans_clustering_section(pca_result, patient_feature_matrix):
    """Sección completa de K-Means: sweep, slider de K, scatter, tamaños y perfil.
    Devuelve (labels, model) para usarlos en validación y análisis de fallas."""
    pca_array = pca_result['pca_df'].drop(columns=['patient_id']).values

    st.markdown("##### Barrido de parámetros (K = 2 a 10)")
    with st.spinner("Calculando inertia y silhouette para cada K..."):
        sweep_results = kmeans_parameter_sweep(pca_array, 2, 10)
    _plot_kmeans_sweep(sweep_results)

    best_row = sweep_results.loc[sweep_results['silhouette'].idxmax()]
    st.caption(
        f"Mejor K según silhouette score: **{int(best_row['k'])}** "
        f"(silhouette = {best_row['silhouette']:.4f})"
    )

    st.divider()

    k_selected = st.slider("Selecciona el número de clusters (K):", min_value=2, max_value=10,
                            value=int(best_row['k']), key="kmeans_k_slider")

    km_model = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
    labels = km_model.fit_predict(pca_array)
    sil_score = silhouette_score(pca_array, labels)

    c1, c2, c3 = st.columns(3)
    c1.metric("K seleccionado", k_selected)
    c2.metric("Silhouette Score", f"{sil_score:.4f}")
    c3.metric("Inertia", f"{km_model.inertia_:,.1f}")

    col_scatter, col_sizes = st.columns([2, 1])
    with col_scatter:
        _plot_kmeans_scatter(pca_array, labels, km_model)
    with col_sizes:
        _plot_cluster_sizes(labels, title=f"Tamaño de clusters (K={k_selected})")

    st.markdown("##### Perfil promedio por cluster")
    df_profile = patient_feature_matrix.copy()
    df_profile['Cluster'] = labels
    key_cols = [c for c in ['age', 'total_admissions', 'avg_billing_amount', 'avg_length_of_stay']
                if c in df_profile.columns]
    profile = df_profile.groupby('Cluster')[key_cols].mean().round(2)
    st.dataframe(profile, width='stretch')

    return labels, km_model


# =======================================================================
# DBSCAN
# =======================================================================
def _plot_k_distance(pca_array, n_neighbors):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(pca_array)
    distances, _ = nn.kneighbors(pca_array)
    distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(distances, color='#5da5da')
    ax.set_title("Gráfico de K-distancia (referencia para elegir eps)", fontsize=13)
    ax.set_xlabel("Puntos ordenados")
    ax.set_ylabel(f"Distancia al {n_neighbors}º vecino")
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


@st.cache_data(show_spinner=False)
def dbscan_parameter_sweep(pca_array, eps_list, min_samples_list):
    rows = []
    for eps in eps_list:
        for min_samples in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(pca_array)
            mask = labels != -1
            distinct = set(labels[mask])
            n_clusters = len(distinct)
            noise = int(np.sum(labels == -1))
            if n_clusters > 1 and mask.sum() > 1:
                score = silhouette_score(pca_array[mask], labels[mask])
            else:
                score = np.nan
            rows.append({"eps": eps, "min_samples": min_samples,
                         "n_clusters": n_clusters, "ruido": noise, "silhouette": score})
    return pd.DataFrame(rows)


def _plot_dbscan_scatter(pca_array, labels):
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(pca_array[:, 0], pca_array[:, 1], c=labels, cmap='viridis',
                          s=40, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_title("Clusters DBSCAN en espacio PCA (PC_1 vs PC_2)", fontsize=13)
    ax.set_xlabel("PC_1")
    ax.set_ylabel("PC_2")
    fig.colorbar(scatter, ax=ax, label="Cluster (-1 = ruido)")
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def dbscan_clustering_section(pca_result, patient_feature_matrix):
    """Sección completa de DBSCAN: k-distance plot, sliders de eps/min_samples,
    barrido de parámetros, scatter final. Devuelve labels."""
    pca_array = pca_result['pca_df'].drop(columns=['patient_id']).values
    n_features = pca_array.shape[1]

    st.markdown("##### Gráfico de K-distancia")
    st.caption("Ayuda a estimar un buen valor de eps: busca el 'codo' de la curva.")
    _plot_k_distance(pca_array, n_neighbors=2 * n_features)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        eps_selected = st.slider("eps:", min_value=0.5, max_value=5.0, value=2.5, step=0.1,
                                  key="dbscan_eps_slider")
    with col2:
        min_samples_selected = st.slider("min_samples:", min_value=2, max_value=30, value=5,
                                          key="dbscan_min_samples_slider")

    db_model = DBSCAN(eps=eps_selected, min_samples=min_samples_selected)
    labels = db_model.fit_predict(pca_array)

    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_ruido = int(np.sum(labels == -1))

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters detectados", n_clusters)
    c2.metric("Puntos de ruido", f"{n_ruido:,}")
    c3.metric("% de ruido", f"{n_ruido / len(labels) * 100:.1f}%")

    if n_clusters > 1 and mask.sum() > 1:
        sil = silhouette_score(pca_array[mask], labels[mask])
        st.caption(f"Silhouette score (excluyendo ruido): **{sil:.4f}**")
    else:
        st.warning("No hay suficientes clusters distintos para calcular silhouette score con estos parámetros.")

    _plot_dbscan_scatter(pca_array, labels)

    with st.expander("Barrido de parámetros (eps × min_samples)"):
        eps_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        min_samples_list = [5, 10, 20]
        with st.spinner("Probando combinaciones de eps y min_samples..."):
            sweep = dbscan_parameter_sweep(pca_array, eps_list, min_samples_list)
        st.dataframe(sweep, width='stretch')

    return labels, db_model


# =======================================================================
# VALIDACIÓN — K-MEANS vs DBSCAN
# =======================================================================
def show_validation_table(pca_result, kmeans_labels, dbscan_labels):
    pca_array = pca_result['pca_df'].drop(columns=['patient_id']).values

    sil_km = silhouette_score(pca_array, kmeans_labels)
    ch_km = calinski_harabasz_score(pca_array, kmeans_labels)
    n_clusters_km = len(set(kmeans_labels))

    db_mask = dbscan_labels != -1
    n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    if n_clusters_db > 1 and db_mask.sum() > 1:
        sil_db = silhouette_score(pca_array[db_mask], dbscan_labels[db_mask])
        ch_db = calinski_harabasz_score(pca_array[db_mask], dbscan_labels[db_mask])
    else:
        sil_db, ch_db = np.nan, np.nan
    noise_db = int(np.sum(dbscan_labels == -1))

    tabla = pd.DataFrame({
        "Métrica / Criterio": [
            "Número de clusters detectados",
            "Coeficiente de Silueta",
            "Índice Calinski-Harabasz",
            "Puntos de ruido (outliers)",
        ],
        "K-Means": [
            n_clusters_km,
            f"{sil_km:.4f}",
            f"{ch_km:.2f}",
            "0 (fuerza a agrupar todo)",
        ],
        "DBSCAN": [
            n_clusters_db,
            f"{sil_db:.4f}" if not np.isnan(sil_db) else "N/A",
            f"{ch_db:.2f}" if not np.isnan(ch_db) else "N/A",
            f"{noise_db} ({noise_db / len(dbscan_labels) * 100:.1f}%)",
        ],
    })

    st.markdown("##### Tabla de validación final")
    st.dataframe(tabla, width='stretch', hide_index=True)


# =======================================================================
# ANÁLISIS DE FALLAS / ANOMALÍAS
# =======================================================================
def show_failure_analysis(pca_result, patient_feature_matrix, kmeans_labels, dbscan_labels):
    pca_array = pca_result['pca_df'].drop(columns=['patient_id']).values

    # --- K-Means: siluetas negativas (casos híbridos) ---
    sil_samples = silhouette_samples(pca_array, kmeans_labels)
    df_km = patient_feature_matrix.copy()
    df_km['Silueta_Individual'] = sil_samples
    fallas_km = df_km[df_km['Silueta_Individual'] < 0]
    pct_fallas_km = len(fallas_km) / len(df_km) * 100 if len(df_km) else 0

    # --- DBSCAN: puntos de ruido ---
    df_db = patient_feature_matrix.copy()
    df_db['Cluster_DBSCAN'] = dbscan_labels
    ruido_db = df_db[df_db['Cluster_DBSCAN'] == -1]
    pct_ruido_db = len(ruido_db) / len(df_db) * 100 if len(df_db) else 0

    st.markdown("##### K-Means — Casos híbridos (silueta negativa)")
    c1, c2 = st.columns(2)
    c1.metric("Observaciones con silueta negativa", f"{len(fallas_km):,}")
    c2.metric("% del total", f"{pct_fallas_km:.2f}%")

    if len(fallas_km) > 0:
        with st.expander("Ver pacientes en conflicto (K-Means)"):
            cols_show = [c for c in ['age', 'total_admissions', 'avg_billing_amount',
                                      'avg_length_of_stay', 'Silueta_Individual'] if c in fallas_km.columns]
            st.dataframe(
                fallas_km[cols_show].sort_values('Silueta_Individual').head(50),
                width='stretch'
            )
    else:
        st.success("No se detectaron casos híbridos con silueta negativa.")

    st.divider()

    st.markdown("##### DBSCAN — Puntos de ruido (casos atípicos)")
    c3, c4 = st.columns(2)
    c3.metric("Puntos de ruido", f"{len(ruido_db):,}")
    c4.metric("% del total", f"{pct_ruido_db:.2f}%")

    if len(ruido_db) > 0:
        with st.expander("Ver pacientes marcados como ruido (DBSCAN)"):
            cols_show = [c for c in ['age', 'total_admissions', 'avg_billing_amount',
                                      'avg_length_of_stay'] if c in ruido_db.columns]
            st.dataframe(ruido_db[cols_show].head(50), width='stretch')
    else:
        st.success("No se detectaron puntos de ruido con estos parámetros.")