import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def contar_outliers_iqr(serie):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (serie < lim_inf) | (serie > lim_sup)
    return mask.sum(), lim_inf, lim_sup


def plot_categorical(df, col):
    col_plot, col_table = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.countplot(data=df, y=col, order=df[col].value_counts().index,
                       hue=col, palette='viridis', legend=False, ax=ax)
        ax.set_title(f'Distribución de {col}', fontsize=13, pad=10)
        ax.set_xlabel('Cantidad')
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with col_table:
        st.markdown(f"##### Conteo — {col}")
        conteo = df[col].value_counts().rename("Cantidad")
        porcentaje = (df[col].value_counts(normalize=True) * 100).round(1).rename("%")
        tabla = pd.concat([conteo, porcentaje], axis=1)
        st.dataframe(tabla, width='stretch')


def plot_numerical(df, col):
    n_outliers, lim_inf, lim_sup = contar_outliers_iqr(df[col])

    col_plot, col_info = st.columns([2, 1])

    with col_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=ax1, color='#5da5da')
        ax1.set_title(f'Histograma de {col}', fontsize=12)

        sns.boxplot(y=df[col], ax=ax2, color='#f17c67')
        ax2.set_title(f'Boxplot de {col}', fontsize=12)

        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    with col_info:
        st.markdown(f"##### Resumen — {col}")
        st.metric("Outliers (IQR)", f"{n_outliers:,}")
        st.caption(f"Límites: {lim_inf:.1f} — {lim_sup:.1f}")

        st.markdown("###### Estadísticas descriptivas")
        stats = df[col].describe().round(2).rename("Valor")
        st.dataframe(stats, width='stretch')

        st.markdown("###### Distribución por rangos")
        bins = pd.cut(df[col], bins=5)
        conteo_bins = bins.value_counts().sort_index().rename("Cantidad")
        st.dataframe(conteo_bins, width='stretch')