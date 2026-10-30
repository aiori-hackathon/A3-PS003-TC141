
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_file = Path("results/fedavg_results.csv")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Could not find {results_file}. "
            "Run this script from the project root."
        )

    df = pd.read_csv(results_file)

    required_columns = {
        "round",
        "mae",
        "rmse",
        "communication_bytes",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    # Plot 1: MAE across rounds
    plt.figure()
    plt.plot(df["round"], df["mae"], marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("MAE")
    plt.title("FedAvg: MAE Across Training Rounds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fedavg_mae.png", dpi=200)
    plt.close()

    # Plot 2: RMSE across rounds
    plt.figure()
    plt.plot(df["round"], df["rmse"], marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("RMSE")
    plt.title("FedAvg: RMSE Across Training Rounds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fedavg_rmse.png", dpi=200)
    plt.close()

    # Plot 3: Communication per round
    plt.figure()
    plt.bar(df["round"], df["communication_bytes"])
    plt.xlabel("Federated Round")
    plt.ylabel("Estimated Communication (bytes)")
    plt.title("FedAvg: Communication Cost Per Round")
    plt.tight_layout()
    plt.savefig(output_dir / "fedavg_communication.png", dpi=200)
    plt.close()

    print("Graphs created successfully:")
    for filename in [
        "fedavg_mae.png",
        "fedavg_rmse.png",
        "fedavg_communication.png",
    ]:
        print(output_dir / filename)


if __name__ == "__main__":
    main()
