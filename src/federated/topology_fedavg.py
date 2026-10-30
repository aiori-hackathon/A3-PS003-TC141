
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.load_dataset import load_dataset
from src.models.traffic_model import TrafficModel


# Experiment configuration
SEED = 42
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001

PARTITION_PATH = Path("results/sensor_partition.json")
OUTPUT_DIR = Path("results/topology_fedavg")


def train_local(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        optimizer.zero_grad()

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(inputs)

    return total_loss / len(loader.dataset)


def average_weights(client_states, client_sizes):
    total_size = sum(client_sizes)
    averaged = {}

    for key in client_states[0]:
        averaged[key] = sum(
            client_states[i][key] * (client_sizes[i] / total_size)
            for i in range(len(client_states))
        )

    return averaged


def evaluate_model(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        predictions = model(X_test).numpy().ravel()

    actual = y_test.numpy().ravel()

    mae = mean_absolute_error(actual, predictions)
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

    return float(mae), rmse


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not PARTITION_PATH.exists():
        raise FileNotFoundError(
            "Sensor partition not found. Run partition_sensors.py first."
        )

    with open(PARTITION_PATH, "r") as file:
        partition = json.load(file)

    X_train, X_test, y_train, y_test, _ = load_dataset()

    # Prepare the complete test set.
    X_test_tensor = torch.from_numpy(
        X_test.reshape(-1, X_test.shape[-1]).astype(np.float32)
    )

    y_test_tensor = torch.from_numpy(
        y_test.reshape(-1, 1).astype(np.float32)
    )

    client_loaders = []
    client_sizes = []

    print("=== TOPOLOGY-AWARE FEDERATED LEARNING ===")

    # Each client receives only its assigned sensors.
    for client in partition["clients"]:
        client_id = client["client_id"]
        sensor_ids = client["sensors"]

        client_x = X_train[:, sensor_ids, :].reshape(
            -1, X_train.shape[-1]
        ).astype(np.float32)

        client_y = y_train[:, sensor_ids].reshape(
            -1, 1
        ).astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(client_x),
            torch.from_numpy(client_y),
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        client_loaders.append(loader)
        client_sizes.append(len(dataset))

        print(
            f"Client {client_id}: "
            f"{len(sensor_ids)} sensors | "
            f"{len(dataset)} examples | "
            f"Sensors: {sensor_ids}"
        )

    num_clients = len(client_loaders)

    global_model = TrafficModel(output_sensors=1)
    loss_fn = nn.MSELoss()

    model_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in global_model.state_dict().values()
    )

    print(f"\nClients: {num_clients}")
    print(f"Rounds: {ROUNDS}")
    print(f"Model size: {model_bytes:,} bytes")

    results = []
    cumulative_communication = 0

    for round_number in range(1, ROUNDS + 1):
        round_start = time.time()

        client_states = []
        local_losses = []

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)

            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=LEARNING_RATE,
            )

            for _ in range(LOCAL_EPOCHS):
                loss = train_local(
                    local_model,
                    loader,
                    optimizer,
                    loss_fn,
                )

            client_states.append({
                key: value.detach().clone()
                for key, value in local_model.state_dict().items()
            })

            local_losses.append(loss)

        # Federated averaging.
        global_model.load_state_dict(
            average_weights(client_states, client_sizes)
        )

        # Evaluate across all 36 sensors.
        mae, rmse = evaluate_model(
            global_model,
            X_test_tensor,
            y_test_tensor,
        )

        # Model download + model update upload.
        round_communication = (
            2 * num_clients * model_bytes
        )

        cumulative_communication += round_communication

        elapsed = time.time() - round_start

        result = {
            "round": round_number,
            "mae": mae,
            "rmse": rmse,
            "mean_local_loss": float(np.mean(local_losses)),
            "round_communication_bytes": round_communication,
            "cumulative_communication_bytes": cumulative_communication,
            "training_seconds": round(elapsed, 2),
        }

        results.append(result)

        print(
            f"Round {round_number:02d}/{ROUNDS} | "
            f"MAE: {mae:.6f} | "
            f"RMSE: {rmse:.6f} | "
            f"Communication: {round_communication:,} bytes | "
            f"Time: {elapsed:.2f}s"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "results.json", "w") as file:
        json.dump(results, file, indent=2)

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "results.csv",
        index=False,
    )

    torch.save(
        global_model.state_dict(),
        OUTPUT_DIR / "global_model.pt",
    )

    summary = {
        "experiment": "Topology-Aware FedAvg",
        "clients": num_clients,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "sensor_partition": partition["clients"],
        "cross_client_edges": partition["total_cut_edges"],
        "final_mae": results[-1]["mae"],
        "final_rmse": results[-1]["rmse"],
        "total_communication_bytes": cumulative_communication,
        "total_training_seconds": round(
            sum(item["training_seconds"] for item in results), 2
        ),
    }

    with open(OUTPUT_DIR / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("\n=== EXPERIMENT COMPLETED ===")
    print(f"Final MAE: {summary['final_mae']:.6f}")
    print(f"Final RMSE: {summary['final_rmse']:.6f}")
    print(
        "Total communication:",
        f"{cumulative_communication:,} bytes",
    )
    print(f"Results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
