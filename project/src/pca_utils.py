import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


# ---------------------------------------------------------------------
# 1. Construcción de la matriz de características por paciente
# ---------------------------------------------------------------------
def build_patient_feature_matrix(df):
    df = df.copy()

    # Identificador de paciente (aproximado, ya que no hay ID real en el dataset)
    df['patient_hash'] = (
        df['Age'].astype(str) + '_' + df['Gender'].astype(str) + '_' + df['Blood Type'].astype(str)
    )
    df['patient_id'] = df.groupby('patient_hash').ngroup()
    df = df.drop(columns=['patient_hash'])

    df['condition_id'] = df['Medical Condition'].astype('category').cat.codes
    df['medication_id'] = df['Medication'].astype('category').cat.codes
    df['record_id'] = range(len(df))

    patients = df[['patient_id', 'Age', 'Gender', 'Blood Type']].drop_duplicates()

    medical_records = df[[
        'record_id', 'patient_id', 'condition_id', 'medication_id',
        'Admission Type', 'Billing Amount', 'Test Results', 'Length of Stay'
    ]]

    patient_medical_data = pd.merge(patients, medical_records, on='patient_id', how='left')

    # Matriz base: estadísticas agregadas por paciente
    patient_feature_matrix = patient_medical_data.groupby('patient_id').agg(
        age=('Age', 'first'),
        total_admissions=('record_id', 'count'),
        avg_billing_amount=('Billing Amount', 'mean'),
        avg_length_of_stay=('Length of Stay', 'mean'),
    ).reset_index()

    # One-hot encoding de Gender y Blood Type
    patient_demographics_encoded = pd.get_dummies(
        patients, columns=['Gender', 'Blood Type'], prefix=['Gender', 'BloodType']
    ).drop(columns=['Age'])

    patient_feature_matrix = pd.merge(
        patient_feature_matrix, patient_demographics_encoded, on='patient_id', how='left'
    )

    # Conteo por condición médica
    condition_counts = patient_medical_data.groupby('patient_id')['condition_id'].value_counts().unstack(fill_value=0)
    condition_counts.columns = [f'condition_{str(c).lower().replace(" ", "_")}_count' for c in condition_counts.columns]

    most_frequent_condition = (
        patient_medical_data.groupby('patient_id')['condition_id']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={'condition_id': 'most_frequent_condition_id'})
    )

    # Conteo por medicamento
    medication_counts = patient_medical_data.groupby('patient_id')['medication_id'].value_counts().unstack(fill_value=0)
    medication_counts.columns = [f'medication_{str(c).lower().replace(" ", "_")}_count' for c in medication_counts.columns]

    most_frequent_medication = (
        patient_medical_data.groupby('patient_id')['medication_id']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={'medication_id': 'most_frequent_medication_id'})
    )

    # Proporción de resultados de pruebas
    test_results_counts = (
        patient_medical_data.groupby('patient_id')['Test Results']
        .value_counts(normalize=True).unstack(fill_value=0)
    )
    test_results_counts.columns = [f'test_result_{str(c).lower().replace(" ", "_")}_prop' for c in test_results_counts.columns]

    # Unión final
    for tabla in [condition_counts, most_frequent_condition, medication_counts,
                  most_frequent_medication, test_results_counts]:
        patient_feature_matrix = pd.merge(patient_feature_matrix, tabla, on='patient_id', how='left')

    return patient_feature_matrix


# ---------------------------------------------------------------------
# 2. PCA
# ---------------------------------------------------------------------
def run_pca(patient_feature_matrix, variance_threshold=0.95):
    features = patient_feature_matrix.drop(
        columns=['patient_id', 'most_frequent_condition_id', 'most_frequent_medication_id']
    )

    numeric_features = features.select_dtypes(include=['number'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    pca_full = PCA()
    pca_full.fit(scaled_features)
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = explained_variance_ratio.cumsum()

    num_components = next(
        i for i, cv in enumerate(cumulative_variance) if cv >= variance_threshold
    ) + 1

    pca_final = PCA(n_components=num_components)
    components = pca_final.fit_transform(scaled_features)

    pca_df = pd.DataFrame(
        data=components,
        columns=[f'PC_{i+1}' for i in range(num_components)]
    )
    pca_df['patient_id'] = patient_feature_matrix['patient_id'].values

    return {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'num_components': num_components,
        'pca_df': pca_df,
        'pca_model': pca_final,
        'feature_names': numeric_features.columns.tolist(),
    }


# ---------------------------------------------------------------------
# 3. Visualizaciones para Streamlit
# ---------------------------------------------------------------------
def plot_variance_explained(pca_result):
    explained_variance_ratio = pca_result['explained_variance_ratio']
    cumulative_variance = pca_result['cumulative_variance']
    num_components = pca_result['num_components']

    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                marker='o', linestyle='--', color='#5da5da')
        ax.axhline(y=0.95, color='#f17c67', linestyle=':', label='Umbral 95%')
        ax.axvline(x=num_components, color='#f17c67', linestyle=':')
        ax.set_title('Varianza Explicada Acumulada por Componente', fontsize=13)
        ax.set_xlabel('Número de Componentes Principales')
        ax.set_ylabel('Varianza Acumulada Explicada')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with col_info:
        st.markdown("##### Resumen PCA")
        st.metric("Componentes para 95% varianza", num_components)
        st.metric("Varianza explicada (top 1)", f"{explained_variance_ratio[0]*100:.1f}%")
        st.metric("Varianza explicada (top 2)", f"{explained_variance_ratio[:2].sum()*100:.1f}%")


def show_components_table(pca_result):
    st.markdown("##### Varianza explicada por componente")
    tabla = pd.DataFrame({
        'Componente': [f'PC_{i+1}' for i in range(len(pca_result['explained_variance_ratio']))],
        'Varianza explicada (%)': (pca_result['explained_variance_ratio'] * 100).round(2),
        'Varianza acumulada (%)': (pca_result['cumulative_variance'] * 100).round(2),
    })
    st.dataframe(tabla, width='stretch')


def plot_pca_scatter(pca_result, patient_feature_matrix):
    pca_df = pca_result['pca_df']

    if pca_result['num_components'] < 2:
        st.warning("Se necesitan al menos 2 componentes para el scatter PC1 vs PC2.")
        return

    color_by = st.selectbox(
        "Colorear por:",
        ["most_frequent_condition_id", "Gender_0", "total_admissions"],
        index=0,
    )

    merged = pd.merge(pca_df, patient_feature_matrix[['patient_id', color_by]], on='patient_id', how='left')

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        merged['PC_1'], merged['PC_2'],
        c=merged[color_by].astype(float), cmap='viridis', alpha=0.7, s=25
    )
    ax.set_title('Pacientes proyectados en PC1 vs PC2', fontsize=13)
    ax.set_xlabel('PC_1')
    ax.set_ylabel('PC_2')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_by)
    fig.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def show_full_matrix(patient_feature_matrix):
    st.markdown("##### Matriz de características completa")
    st.caption(f"Dimensiones: {patient_feature_matrix.shape[0]} pacientes × {patient_feature_matrix.shape[1]} columnas")
    st.dataframe(patient_feature_matrix, width='stretch')