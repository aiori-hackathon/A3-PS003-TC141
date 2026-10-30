
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


SEED = 42
NUM_CLIENTS = 3
ROUNDS = 10
LOCAL_EPOCHS = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001


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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, adjacency = load_dataset()

    # Test data: one example per sensor per time sample.
    X_test = X_test.reshape(-1, X_test.shape[-1]).astype(np.float32)
    y_test = y_test.reshape(-1, 1).astype(np.float32)

    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # Assign fixed groups of sensors to three clients.
    sensor_count = X_train.shape[1]
    sensor_groups = np.array_split(
        np.arange(sensor_count), NUM_CLIENTS
    )

    client_loaders = []
    client_sizes = []

    for client_id, sensor_ids in enumerate(sensor_groups, start=1):
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

        client_loaders.append(
            DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        )
        client_sizes.append(len(dataset))

        print(
            f"Client {client_id}: sensors "
            f"{sensor_ids[0]}–{sensor_ids[-1]}, "
            f"{len(sensor_ids)} sensors, {len(dataset)} examples"
        )

    # Initialise one global model for the entire experiment.
    global_model = TrafficModel(output_sensors=1)
    loss_fn = nn.MSELoss()
    results = []

    model_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in global_model.state_dict().values()
    )

    print("\n=== FEDERATED LEARNING: FedAvg ===")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Training examples per client: {client_sizes}")
    print(f"Rounds: {ROUNDS}")
    print(f"Model update size: {model_bytes:,} bytes")

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
                    local_model, loader, optimizer, loss_fn
                )

            client_states.append({
                key: value.detach().clone()
                for key, value in local_model.state_dict().items()
            })
            local_losses.append(loss)

        global_model.load_state_dict(
            average_weights(client_states, client_sizes)
        )

        global_model.eval()
        with torch.no_grad():
            predictions = global_model(X_test).numpy().ravel()

        actual = y_test.numpy().ravel()
        mae = mean_absolute_error(actual, predictions)
        rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

        # Estimated model payload: download plus upload for each client.
        communication_bytes = 2 * NUM_CLIENTS * model_bytes
        elapsed = time.time() - round_start

        results.append({
            "round": round_number,
            "mae": float(mae),
            "rmse": rmse,
            "mean_local_loss": float(np.mean(local_losses)),
            "communication_bytes": communication_bytes,
            "training_seconds": round(elapsed, 2),
        })

        print(
            f"Round {round_number:02d}/{ROUNDS} | "
            f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | "
            f"Communication: {communication_bytes:,} bytes | "
            f"Time: {elapsed:.2f}s"
        )

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "fedavg_results.json", "w") as file:
        json.dump(results, file, indent=2)

    pd.DataFrame(results).to_csv(
        output_dir / "fedavg_results.csv", index=False
    )

    torch.save(
        global_model.state_dict(),
        output_dir / "fedavg_global_model.pt",
    )

    print("\nFederated experiment completed.")
    print("Saved results to results/fedavg_results.csv and .json")


if __name__ == "__main__":
    main()
