
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np

from src.data.load_dataset import load_dataset


NUM_CLIENTS = 3
SENSORS_PER_CLIENT = 12
TRIALS = 5000
SEED = 42


def create_partition(graph, rng):
    """Grow three connected groups of equal size."""
    nodes = list(graph.nodes())
    seeds = rng.sample(nodes, NUM_CLIENTS)

    groups = [{seed} for seed in seeds]
    assigned = set(seeds)

    while len(assigned) < len(nodes):
        eligible = []

        for client_id, group in enumerate(groups):
            if len(group) >= SENSORS_PER_CLIENT:
                continue

            frontier = set()
            for node in group:
                frontier.update(graph.neighbors(node))

            frontier -= assigned

            if frontier:
                eligible.append((client_id, list(frontier)))

        if not eligible:
            return None

        client_id, frontier = rng.choice(eligible)
        selected = rng.choice(frontier)

        groups[client_id].add(selected)
        assigned.add(selected)

    return groups


def count_cut_edges(graph, groups):
    """Count edges connecting different clients."""
    owner = {}

    for client_id, group in enumerate(groups):
        for node in group:
            owner[node] = client_id

    return sum(
        1
        for u, v in graph.edges()
        if owner[u] != owner[v]
    )


def main():
    _, _, _, _, adjacency = load_dataset()

    graph = nx.from_numpy_array(adjacency)

    if not nx.is_connected(graph):
        raise ValueError("Sensor network is not fully connected.")

    if graph.number_of_nodes() != NUM_CLIENTS * SENSORS_PER_CLIENT:
        raise ValueError("Network size does not match partition settings.")

    rng = random.Random(SEED)
    best_groups = None
    best_cut_edges = float("inf")

    for _ in range(TRIALS):
        groups = create_partition(graph, rng)

        if groups is None:
            continue

        cut_edges = count_cut_edges(graph, groups)

        if cut_edges < best_cut_edges:
            best_cut_edges = cut_edges
            best_groups = groups

    if best_groups is None:
        raise RuntimeError("Could not find a valid partition.")

    results = {
        "algorithm": "Randomized Connected Region Growth",
        "clients": [],
        "total_cut_edges": int(best_cut_edges),
        "total_edges": graph.number_of_edges(),
        "trials": TRIALS,
        "seed": SEED,
    }

    for client_id, group in enumerate(best_groups, start=1):
        subgraph = graph.subgraph(group)

        results["clients"].append({
            "client_id": client_id,
            "sensors": sorted(int(node) for node in group),
            "sensor_count": len(group),
            "internal_edges": subgraph.number_of_edges(),
            "connected": nx.is_connected(subgraph),
        })

    output_path = Path("results/sensor_partition.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)

    print("\n=== SENSOR NETWORK PARTITION ===")

    for client in results["clients"]:
        print(
            f"Client {client['client_id']}: "
            f"{client['sensors']}"
        )
        print(
            f"  Sensors: {client['sensor_count']} | "
            f"Internal edges: {client['internal_edges']} | "
            f"Connected: {client['connected']}"
        )

    print(f"\nTotal network edges: {graph.number_of_edges()}")
    print(f"Connections crossing clients: {best_cut_edges}")
    print(f"Trials performed: {TRIALS}")
    print(f"Saved partition to: {output_path}")


if __name__ == "__main__":
    main()
