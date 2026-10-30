
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path("results")
OUTPUT = RESULTS / "final_analysis"


def read_json(path):
    with open(path, "r") as file:
        return json.load(file)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    centralized = read_json(
        RESULTS / "centralized_neural_network.json"
    )

    local = read_json(
        RESULTS / "local_only_results.json"
    )

    random_fedavg = read_json(
        RESULTS / "fedavg_results.json"
    )

    topology = read_json(
        RESULTS / "topology_fedavg" / "summary.json"
    )

    topology_rounds = read_json(
        RESULTS / "topology_fedavg" / "results.json"
    )

    random_time = sum(
        item["training_seconds"] for item in random_fedavg
    )

    topology_time = topology["total_training_seconds"]

    comparison = pd.DataFrame([
        {
            "model": "Centralized NN",
            "mae": centralized["mae"],
            "rmse": centralized["rmse"],
            "training_seconds": centralized["training_seconds"],
            "communication_bytes": None,
        },
        {
            "model": "Local-only",
            "mae": local["overall_mae"],
            "rmse": local["overall_rmse"],
            "training_seconds": local["total_training_seconds"],
            "communication_bytes": 0,
        },
        {
            "model": "Standard FedAvg",
            "mae": random_fedavg[-1]["mae"],
            "rmse": random_fedavg[-1]["rmse"],
            "training_seconds": round(random_time, 2),
            "communication_bytes": sum(
                item["communication_bytes"]
                for item in random_fedavg
            ),
        },
        {
            "model": "Topology FedAvg",
            "mae": topology["final_mae"],
            "rmse": topology["final_rmse"],
            "training_seconds": topology_time,
            "communication_bytes": topology[
                "total_communication_bytes"
            ],
        },
    ])

    comparison.to_csv(
        OUTPUT / "final_model_comparison.csv",
        index=False,
    )

    print("\n=== FINAL MODEL COMPARISON ===")
    print(comparison.to_string(index=False))

    metrics = [
        ("mae", "Mean Absolute Error", "comparison_mae.png"),
        ("rmse", "Root Mean Squared Error", "comparison_rmse.png"),
        (
            "training_seconds",
            "Training Time (seconds)",
            "comparison_training_time.png",
        ),
    ]

    for metric, title, filename in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(comparison["model"], comparison[metric])
        plt.title(title)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT / filename, dpi=300)
        plt.close()

    # Compare federated convergence.
    plt.figure(figsize=(10, 6))

    plt.plot(
        [item["round"] for item in random_fedavg],
        [item["mae"] for item in random_fedavg],
        marker="o",
        label="Standard FedAvg",
    )

    plt.plot(
        [item["round"] for item in topology_rounds],
        [item["mae"] for item in topology_rounds],
        marker="o",
        label="Topology FedAvg",
    )

    plt.xlabel("Federated Round")
    plt.ylabel("MAE")
    plt.title("Federated Learning Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "federated_convergence.png", dpi=300)
    plt.close()

    # Compare cumulative communication.
    plt.figure(figsize=(10, 6))

    plt.plot(
        [item["round"] for item in random_fedavg],
        [
            sum(
                previous["communication_bytes"]
                for previous in random_fedavg[:i]
            )
            / 1024
            for i in range(1, len(random_fedavg) + 1)
        ],
        marker="o",
        label="Standard FedAvg",
    )

    plt.plot(
        [item["round"] for item in topology_rounds],
        [
            item["cumulative_communication_bytes"] / 1024
            for item in topology_rounds
        ],
        marker="o",
        label="Topology FedAvg",
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Cumulative Communication (KB)")
    plt.title("Communication Cost Across Training Rounds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "communication_comparison.png", dpi=300)
    plt.close()

    print("\nAnalysis completed.")
    print("Saved comparison table and five graphs to:", OUTPUT)


if __name__ == "__main__":
    main()
