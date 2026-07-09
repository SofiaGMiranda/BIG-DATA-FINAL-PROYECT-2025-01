import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

LABEL_MAPS = {
    'Medical Condition': {0:'Arthritis',1:'Asthma',2:'Cancer',
                          3:'Diabetes',4:'Hypertension',5:'Obesity'},
    'Medication':        {0:'Aspirin',1:'Ibuprofen',2:'Lipitor',
                          3:'Paracetamol',4:'Penicillin'},
}

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CONDITION_PATH = BASE_DIR.parent / "data" / "interim" / "healtcare_processedv2.csv"
CLUSTERS_PATH = BASE_DIR.parent / "data" / "processed" / "healthcare_pca_clusters.csv"

#st.write("BASE_DIR:", BASE_DIR)
#st.write("CONDITION_PATH:", CONDITION_PATH)
#st.write("Exists CONDITION:", CONDITION_PATH.exists())

#st.write("CLUSTERS_PATH:", CLUSTERS_PATH)

#st.write("Exists CLUSTERS:", CLUSTERS_PATH.exists())

df = pd.read_csv(CONDITION_PATH)
@st.cache_data
def load_clinical_data():
    """Loads and prepares the clinical data (same CSVs as Prioritization)."""
    df = pd.read_csv(CONDITION_PATH)
    clusters = pd.read_csv(CLUSTERS_PATH)

    clusters = clusters.dropna(subset=['cluster'])
    clusters['Cluster_Biz'] = clusters['cluster'].astype(int)
    df = df.loc[clusters.index].copy()
    df['Cluster_Biz'] = clusters['Cluster_Biz'].values
    df['Cost_Per_Day'] = df['Billing Amount'] / df['Length of Stay']
    df = df.dropna(subset=['Medical Condition', 'Medication', 'Test Results'])

    df['condition_label'] = df['Medical Condition'].map(LABEL_MAPS['Medical Condition'])
    df['medication_label'] = df['Medication'].map(LABEL_MAPS['Medication'])

    return df


@st.cache_data
def compute_edge_stats(df):
    """Computes weight and lift for each (condition, medication) pair."""
    edge_stats = df.groupby(['condition_label', 'medication_label']).agg(
        n_patients=('Test Results', 'count'),
        n_abnormal=('Test Results', lambda x: (x == 2).sum()),
    ).reset_index()

    edge_stats['prop_abnormal'] = (edge_stats['n_abnormal'] / edge_stats['n_patients']).round(3)
    edge_stats['weight'] = (edge_stats['n_patients'] * (1 + edge_stats['prop_abnormal'])).round(2)

    n_total = len(df)
    p_condition = df['condition_label'].value_counts(normalize=True)
    p_medication = df['medication_label'].value_counts(normalize=True)

    edge_stats['expected_patients'] = edge_stats.apply(
        lambda r: p_condition[r['condition_label']] * p_medication[r['medication_label']] * n_total,
        axis=1
    )
    edge_stats['lift'] = edge_stats['n_patients'] / edge_stats['expected_patients']

    return edge_stats


def build_graph(df, edge_stats, lift_threshold, weight_threshold):
    """Builds the graph applying both filters: lift (significance) and weight (intensity)."""
    conditions = df['condition_label'].unique().tolist()
    medications = df['medication_label'].unique().tolist()

    G = nx.Graph()
    for c in conditions:
        G.add_node(c, node_type='condition')
    for m in medications:
        G.add_node(m, node_type='medication')

    filtered = edge_stats[
        (edge_stats['lift'] >= lift_threshold) &
        (edge_stats['weight'] >= weight_threshold)
    ]

    for _, row in filtered.iterrows():
        G.add_edge(
            row['condition_label'], row['medication_label'],
            weight=row['weight'], lift=round(row['lift'], 3),
            n_patients=int(row['n_patients']), prop_abnormal=row['prop_abnormal'],
        )
    return G



def plot_graph_plotly(G):
    """
    Draws the Condition ↔ Medication graph with Plotly.
    Isolated nodes appear in red.
    """

    isolated = set(nx.isolates(G))

    # -------------------------
    # Positions
    # -------------------------
    pos = {}

    conditions = sorted([
        n for n, d in G.nodes(data=True)
        if d["node_type"] == "condition"
    ])

    medications = sorted([
        n for n, d in G.nodes(data=True)
        if d["node_type"] == "medication"
    ])

    spacing = 2.3

    for i, c in enumerate(conditions):
        pos[c] = (0, i * spacing)

    for i, m in enumerate(medications):
        pos[m] = (6, i * spacing)

    # -------------------------
    # Edges
    # -------------------------
    edge_x = []
    edge_y = []

    for u, v, d in G.edges(data=True):

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(
            width=2,
            color="rgba(120,120,120,0.45)"
        )
    )

    # -------------------------
    # Nodes
    # -------------------------
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    hover_text = []

    for n, d in G.nodes(data=True):

        x, y = pos[n]

        node_x.append(x)
        node_y.append(y)

        grado = G.degree(n)

        node_size.append(20 + grado * 5)

        if n in isolated:
            node_color.append("#E63946")
        elif d["node_type"] == "condition":
            node_color.append("#185FA5")
        else:
            node_color.append("#1D9E75")

        tipo = (
            "Condition"
            if d["node_type"] == "condition"
            else "Medication"
        )

        hover_text.append(
            f"<b>{n}</b><br>"
            f"Type: {tipo}<br>"
            f"Connections: {grado}"
            f"{'<br><b>Isolated node</b>' if n in isolated else ''}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        hovertext=hover_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(
                width=2,
                color="white"
            )
        )
    )

    # -------------------------
    # Labels
    # -------------------------
    label_x = []
    label_y = []
    labels = []

    for n in G.nodes():

        x, y = pos[n]

        if G.nodes[n]["node_type"] == "condition":
            label_x.append(x - 0.25)
            label_y.append(y)
        else:
            label_x.append(x + 0.25)
            label_y.append(y)

        labels.append(n)

    label_trace = go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=labels,
        hoverinfo="skip",
        textfont=dict(
            size=14,
            color="#222",
            family="Arial"
        )
    )

    # -------------------------
    # Figure
    # -------------------------
    fig = go.Figure(
        data=[
            edge_trace,
            node_trace,
            label_trace
        ]
    )

    fig.update_layout(

        template="plotly_white",

        showlegend=False,

        height=650,

        margin=dict(
            l=80,
            r=80,
            t=60,
            b=30
        ),

        plot_bgcolor="white",
        paper_bgcolor="white",

        hovermode="closest",

        xaxis=dict(
            visible=False,
            fixedrange=True
        ),

        yaxis=dict(
            visible=False,
            fixedrange=True
        )
    )

    # -------------------------
    # Column titles
    # -------------------------
    ymax = max(node_y) + spacing

    fig.add_annotation(
        x=0,
        y=ymax,
        text="<b>Conditions</b>",
        showarrow=False,
        font=dict(size=18, color="#185FA5")
    )

    fig.add_annotation(
        x=6,
        y=ymax,
        text="<b>Medications</b>",
        showarrow=False,
        font=dict(size=18, color="#1D9E75")
    )

    return fig, isolated

def run_graph_section():
    """Complete 'Clinical Graph' tab section for app.py"""
    st.subheader("Clinical Graph — Condition ↔ Medication")
    st.caption(
        "Edges represent statistically significant associations "
        "(lift), not simple co-occurrence. Adjust the thresholds to explore "
        "how robust each node's centrality is."
    )

    df = load_clinical_data()
    edge_stats = compute_edge_stats(df)

    col1, col2 = st.columns(2)
    with col1:
        lift_thr = st.slider(
            "Significance threshold (lift)",
            min_value=float(edge_stats['lift'].min()),
            max_value=float(edge_stats['lift'].max()),
            value=1.05, step=0.01,
            help="lift >= 1 means the association occurs more often than expected by chance."
        )
    with col2:
        weight_thr = st.slider(
            "Intensity threshold (weight)",
            min_value=float(edge_stats['weight'].min()),
            max_value=float(edge_stats['weight'].max()),
            value=float(edge_stats['weight'].min()), step=1.0,
            help="Filters by patient volume weighted by severity."
        )

    G = build_graph(df, edge_stats, lift_thr, weight_thr)
    fig, isolated = plot_graph_plotly(G)

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Edges", G.number_of_edges())
    c2.metric("Density", f"{nx.density(G):.3f}")
    c3.metric("Isolated Nodes", len(isolated))

    st.plotly_chart(fig, width='stretch')

    if isolated:
        st.warning(
            f"⚠️ With these thresholds, **{', '.join(sorted(isolated))}** "
            f"become(s) disconnected — its/their connection(s) are statistically "
            f"valid but of lower intensity than the rest of the network."
        )

    with st.expander("View centrality metrics"):
        if nx.is_connected(G) and G.number_of_edges() > 0:
            G_dist = G.copy()
            for u, v, d in G_dist.edges(data=True):
                d['distance'] = 1.0 / d['weight']
            metrics_df = pd.DataFrame({
                'degree': dict(G.degree()),
                'betweenness': nx.betweenness_centrality(G_dist, weight='distance'),
                'closeness': nx.closeness_centrality(G),
                'pagerank': nx.pagerank(G, weight='weight', alpha=0.85),
            }).sort_values('pagerank', ascending=False)
            st.dataframe(metrics_df.round(4), width='stretch')
        else:
            st.info("The graph is not connected with these thresholds — some metrics are not comparable.")