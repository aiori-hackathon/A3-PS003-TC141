
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from scipy.io import loadmat


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="FedTraffic Lab",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

DARK = "#0B1120"
PANEL = "#131D2E"
ACCENT = "#38BDF8"
MUTED = "#94A3B8"


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_experiments():
    centralized = read_json(
        RESULTS / "centralized_neural_network.json"
    )
    local = read_json(
        RESULTS / "local_only_results.json"
    )
    random_rounds = read_json(
        RESULTS / "fedavg_results.json"
    )
    topology_summary = read_json(
        RESULTS / "topology_fedavg" / "summary.json"
    )
    topology_rounds = read_json(
        RESULTS / "topology_fedavg" / "results.json"
    )
    partition = read_json(
        RESULTS / "sensor_partition.json"
    )
    comparison = pd.read_csv(
        RESULTS / "final_analysis" / "final_model_comparison.csv"
    )

    return (
        centralized,
        local,
        random_rounds,
        topology_summary,
        topology_rounds,
        partition,
        comparison,
    )


@st.cache_data
def load_network():
    data = loadmat( ROOT / "data" / "raw" / "traffic_dataset.mat")
    return np.asarray(data["tra_adj_mat"])


(
    centralized,
    local,
    random_rounds,
    topology_summary,
    topology_rounds,
    partition,
    comparison,
) = load_experiments()

adjacency = load_network()


# ---------------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: #0B1120;
        color: #E2E8F0;
    }

    [data-testid="stSidebar"] {
        background: #101A2B;
        border-right: 1px solid #263449;
    }

    [data-testid="stMetric"] {
        background: #131D2E;
        border: 1px solid #263449;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC;
    }

    h1, h2, h3 {
        color: #F8FAFC;
    }

    .hero {
        background: linear-gradient(110deg, #14243B, #102D43);
        border: 1px solid #28445F;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
    }

    .hero-label {
        color: #38BDF8;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 700;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 750;
        color: #F8FAFC;
        margin-top: 8px;
    }

    .hero-description {
        color: #B6C5D8;
        font-size: 15px;
    }

    .section-caption {
        color: #94A3B8;
        font-size: 13px;
    }

    div[data-testid="stPlotlyChart"] {
        background: #131D2E;
        border: 1px solid #263449;
        border-radius: 12px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌐 FedTraffic Lab")
    st.caption("FEDERATED TRAFFIC INTELLIGENCE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Sensor Network",
            "Federated Training",
            "Model Comparison",
            "Communication Analysis",
        ],
    )

    st.divider()

    st.markdown("**EXPERIMENT CONFIGURATION**")
    st.caption("36 sensors")
    st.caption("3 federated clients")
    st.caption("10 communication rounds")
    st.caption("FedAvg aggregation")

    st.divider()

    st.caption("AIORI 3 · PS-003")
    st.caption("TEAM RESONANCE")


# ---------------------------------------------------------
# CHART THEME
# ---------------------------------------------------------

def style_chart(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color="#DCE6F2"),
        height=height,
        margin=dict(l=25, r=25, t=45, b=35),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.12,
        ),
    )
    fig.update_xaxes(gridcolor="#263449", zerolinecolor="#263449")
    fig.update_yaxes(gridcolor="#263449", zerolinecolor="#263449")
    return fig


def page_header(title, description):
    st.title(title)
    st.markdown(
        f'<p class="section-caption">{description}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <div class="hero-label">AIORI 3 · PS-003 · TEAM RESONANCE</div>
        <div class="hero-title">Federated Traffic Intelligence</div>
        <p class="hero-description">
            A communication-aware research platform exploring
            traffic prediction across distributed sensor networks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Experiment at a glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Network Sensors", "36")
    c2.metric("Federated Clients", "3")
    c3.metric("Training Rounds", "10")
    c4.metric("Best MAE", f"{local['overall_mae']:.5f}")

    st.divider()

    st.subheader("Model performance")

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            comparison,
            x="model",
            y="mae",
            color="model",
            title="Prediction Error · MAE",
            labels={"model": "Approach", "mae": "MAE"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        fig = px.bar(
            comparison,
            x="model",
            y="rmse",
            color="model",
            title="Prediction Error · RMSE",
            labels={"model": "Approach", "rmse": "RMSE"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()

    st.subheader("Research snapshot")

    best_mae = comparison.loc[comparison["mae"].idxmin()]
    best_rmse = comparison.loc[comparison["rmse"].idxmin()]

    a, b = st.columns(2)

    with a:
        st.info(
            f"**Lowest MAE:** {best_mae['model']} "
            f"({best_mae['mae']:.6f})"
        )

    with b:
        st.info(
            f"**Lowest RMSE:** {best_rmse['model']} "
            f"({best_rmse['rmse']:.6f})"
        )

    st.caption(
        "Metrics are calculated from the saved experiment results. "
        "Training schedules differ between approaches."
    )


# ---------------------------------------------------------
# SENSOR NETWORK
# ---------------------------------------------------------

elif page == "Sensor Network":

    page_header(
        "Sensor Network Topology",
        "Explore the 36-node traffic network and its topology-aware client assignments.",
    )

    graph = nx.from_numpy_array(adjacency)
    positions = nx.spring_layout(graph, seed=42, k=0.8)

    client_map = {}
    for client in partition["clients"]:
        for sensor in client["sensors"]:
            client_map[sensor] = client["client_id"]

    edge_x, edge_y = [], []
    cross_x, cross_y = [], []

    for u, v in graph.edges():
        x0, y0 = positions[u]
        x1, y1 = positions[v]

        if client_map.get(u) != client_map.get(v):
            cross_x.extend([x0, x1, None])
            cross_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="#64748B", width=1.5),
        hoverinfo="none",
        name="Internal connections",
    ))

    fig.add_trace(go.Scatter(
        x=cross_x,
        y=cross_y,
        mode="lines",
        line=dict(color="#F87171", width=3),
        hoverinfo="none",
        name="Cross-client connections",
    ))

    colors = {
        1: "#38BDF8",
        2: "#FBBF24",
        3: "#4ADE80",
    }

    for client_id in range(1, 4):
        sensors = [
            node for node in graph.nodes()
            if client_map.get(node) == client_id
        ]

        fig.add_trace(go.Scatter(
            x=[positions[n][0] for n in sensors],
            y=[positions[n][1] for n in sensors],
            mode="markers+text",
            text=[str(n) for n in sensors],
            textposition="middle center",
            marker=dict(
                size=27,
                color=colors[client_id],
                line=dict(color="#E2E8F0", width=1),
            ),
            name=f"Client {client_id}",
            hovertemplate=(
                "Sensor %{text}<br>"
                f"Client {client_id}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="36-Sensor Network · Client Ownership",
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(
        style_chart(fig, 650),
        use_container_width=True,
    )

    st.markdown("### Client assignments")

    cols = st.columns(3)

    for i, client in enumerate(partition["clients"]):
        with cols[i]:
            st.markdown(f"**Client {client['client_id']}**")
            st.metric("Sensors", client["sensor_count"])
            st.metric("Internal edges", client["internal_edges"])
            st.caption(
                "Connected region: "
                + ("Yes" if client["connected"] else "No")
            )
            st.write(client["sensors"])

    st.divider()

    a, b, c = st.columns(3)
    a.metric("Total Network Edges", partition["total_edges"])
    b.metric("Cross-client Edges", partition["total_cut_edges"])
    c.metric("Partition Trials", f"{partition['trials']:,}")


# ---------------------------------------------------------
# FEDERATED TRAINING
# ---------------------------------------------------------

elif page == "Federated Training":

    page_header(
        "Federated Training",
        "Inspect how model performance evolves over communication rounds.",
    )

    mode = st.selectbox(
        "Select federated experiment",
        ["Standard FedAvg", "Topology-Aware FedAvg"],
    )

    if mode == "Standard FedAvg":
        rounds = pd.DataFrame(random_rounds)
    else:
        rounds = pd.DataFrame(topology_rounds)

    st.subheader(f"{mode} · Training Progress")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rounds["round"],
        y=rounds["mae"],
        mode="lines+markers",
        name="MAE",
        line=dict(color="#38BDF8", width=3),
    ))

    fig.add_trace(go.Scatter(
        x=rounds["round"],
        y=rounds["rmse"],
        mode="lines+markers",
        name="RMSE",
        line=dict(color="#FBBF24", width=3),
    ))

    fig.update_layout(
        title="Prediction Error Across Training Rounds",
        xaxis_title="Federated Round",
        yaxis_title="Error",
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Round-level experiment data")

    st.dataframe(
        rounds,
        use_container_width=True,
        hide_index=True,
    )

    if mode == "Topology-Aware FedAvg":
        st.info(
            "Clients are assigned connected groups of sensors "
            "using the saved topology-aware partition."
        )
    else:
        st.info(
            "Training examples are randomly distributed across clients."
        )


# ---------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------

elif page == "Model Comparison":

    page_header(
        "Model Comparison",
        "Compare centralized, independent local and federated learning approaches.",
    )

    st.subheader("Experiment metrics")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    metric = st.segmented_control(
        "Comparison metric",
        ["MAE", "RMSE", "Training Time"],
        default="MAE",
    )

    metric_columns = {
        "MAE": ("mae", "Mean Absolute Error"),
        "RMSE": ("rmse", "Root Mean Squared Error"),
        "Training Time": ("training_seconds", "Training Time (seconds)"),
    }

    column, label = metric_columns[metric]

    fig = px.bar(
        comparison,
        x="model",
        y=column,
        color="model",
        title=label,
        text_auto=".4f",
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.caption(
        "Lower values are better for all three displayed metrics. "
        "Training times are measured locally and are hardware-dependent."
    )


# ---------------------------------------------------------
# COMMUNICATION ANALYSIS
# ---------------------------------------------------------

elif page == "Communication Analysis":

    page_header(
        "Communication Analysis",
        "Understand the estimated model-payload cost of federated collaboration.",
    )

    random_total = sum(
        item.get("round_communication_bytes", item.get("communication_bytes", 0))
        for item in random_rounds
    )

    topology_total = topology_summary["total_communication_bytes"]

    a, b, c = st.columns(3)

    a.metric(
        "Standard FedAvg",
        f"{random_total / 1024:.1f} KB",
    )

    b.metric(
        "Topology FedAvg",
        f"{topology_total / 1024:.1f} KB",
    )

    c.metric(
        "Model Update Size",
        "29.44 KB",
    )

    st.divider()

    st.subheader("Cumulative communication")

    random_cumulative = np.cumsum([
        item.get("round_communication_bytes", item.get("communication_bytes", 0))
        for item in random_rounds
    ])

    topology_cumulative = np.cumsum([
        item.get("round_communication_bytes", item.get("communication_bytes", 0))
        for item in topology_rounds
    ])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(random_cumulative) + 1)),
        y=random_cumulative / 1024,
        mode="lines+markers",
        name="Standard FedAvg",
    ))

    fig.add_trace(go.Scatter(
        x=list(range(1, len(topology_cumulative) + 1)),
        y=topology_cumulative / 1024,
        mode="lines+markers",
        name="Topology FedAvg",
    ))

    fig.update_layout(
        title="Cumulative Model-Payload Communication",
        xaxis_title="Federated Round",
        yaxis_title="Communication (KB)",
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()

    st.subheader("Communication methodology")

    st.markdown("""
    Communication is estimated using model-payload accounting:

    - Each client downloads the global model.
    - Each client uploads its model update.
    - One round is estimated as:

    `2 × number of clients × model size`

    This estimate excludes network protocol headers, encryption,
    retries, compression and other transport overhead.
    """)

    st.warning(
        "These are estimated payload bytes, not measurements from "
        "actual network transmission."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "FedTraffic Lab · Team RESONANCE · AIORI 3 · PS-003"
)
