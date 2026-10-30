
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from src.data.load_dataset import load_dataset


def main():
    adjacency = load_dataset()[-1]

    graph = nx.from_numpy_array(adjacency)

    with open("results/sensor_partition.json", "r") as file:
        partition = json.load(file)

    colors = ["skyblue", "orange", "lightgreen"]

    sensor_owner = {}

    for client in partition["clients"]:
        client_id = client["client_id"]

        for sensor in client["sensors"]:
            sensor_owner[sensor] = client_id

    node_colors = [
        colors[sensor_owner[node] - 1]
        for node in graph.nodes()
    ]

    internal_edges = []
    cross_edges = []

    for u, v in graph.edges():
        if sensor_owner[u] == sensor_owner[v]:
            internal_edges.append((u, v))
        else:
            cross_edges.append((u, v))

    position = nx.kamada_kawai_layout(graph)

    plt.figure(figsize=(14, 10))

    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=internal_edges,
        edge_color="gray",
        width=1.5,
        alpha=0.7,
    )

    nx.draw_networkx_edges(
        graph,
        position,
        edgelist=cross_edges,
        edge_color="red",
        width=2.5,
    )

    nx.draw_networkx_nodes(
        graph,
        position,
        node_color=node_colors,
        node_size=850,
        edgecolors="black",
    )

    nx.draw_networkx_labels(
        graph,
        position,
        font_size=10,
    )

    for i, color in enumerate(colors, start=1):
        plt.scatter(
            [], [],
            color=color,
            label=f"Client {i}",
            s=120,
        )

    plt.plot(
        [], [],
        color="red",
        linewidth=2.5,
        label=f"Cross-client edges ({len(cross_edges)})",
    )

    plt.title("FedTraffic Lab - Topology-Aware Sensor Partitioning")
    plt.legend(loc="upper left")
    plt.axis("off")
    plt.tight_layout()

    output = Path("results/plots")
    output.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output / "topology_aware_partition.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved topology-aware partition visualization.")
    print("Client groups:", len(partition["clients"]))
    print("Internal edges:", len(internal_edges))
    print("Cross-client edges:", len(cross_edges))


if __name__ == "__main__":
    main()
