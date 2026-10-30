
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
PLOT_DIR = RESULTS_DIR / "comparison_plots"


def load_json(filename):
    with open(RESULTS_DIR / filename, "r") as file:
        return json.load(file)


def save_comparison_chart(df, metric, title, ylabel, filename):
    plt.figure(figsize=(8, 5))

    plt.bar(df["model"], df[metric])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    centralized = load_json("centralized_neural_network.json")
    local = load_json("local_only_results.json")
    fedavg = load_json("fedavg_results.json")

    # Use the final FedAvg communication round.
    final_fedavg = fedavg[-1]

    # Sum the communication and round times across all rounds.
    total_communication = sum(
        item["communication_bytes"] for item in fedavg
    )
    fedavg_training_time = sum(
        item["training_seconds"] for item in fedavg
    )

    comparison = [
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
            "model": "FedAvg",
            "mae": final_fedavg["mae"],
            "rmse": final_fedavg["rmse"],
            "training_seconds": round(fedavg_training_time, 2),
            "communication_bytes": total_communication,
        },
    ]

    df = pd.DataFrame(comparison)

    # Save combined numerical results.
    df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    # Generate individual comparison plots.
    save_comparison_chart(
        df,
        "mae",
        "Model Comparison - MAE",
        "Mean Absolute Error",
        "comparison_mae.png",
    )

    save_comparison_chart(
        df,
        "rmse",
        "Model Comparison - RMSE",
        "Root Mean Squared Error",
        "comparison_rmse.png",
    )

    save_comparison_chart(
        df,
        "training_seconds",
        "Model Comparison - Training Time",
        "Training Time (seconds)",
        "comparison_training_time.png",
    )

    print("\n=== MODEL COMPARISON ===")
    print(
        df[["model", "mae", "rmse", "training_seconds"]]
        .to_string(index=False)
    )

    print("\n=== COMMUNICATION COST ===")
    print(f"FedAvg total: {total_communication:,} bytes")
    print(f"FedAvg total: {total_communication / 1024:.2f} KB")

    print("\nSaved comparison CSV:")
    print(RESULTS_DIR / "model_comparison.csv")

    print("\nSaved comparison graphs:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()
