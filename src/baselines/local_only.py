
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
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001


def train_model(model, loader, optimizer, loss_fn):
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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, _ = load_dataset()

    partition_path = Path("results/sensor_partition.json")

    with open(partition_path, "r") as file:
        partition = json.load(file)

    sensor_groups = [
        np.array(client["sensors"], dtype=int)
        for client in partition["clients"]
    ]

    NUM_CLIENTS = len(sensor_groups)

    all_predictions = []
    all_actual = []
    client_results = []
    total_training_time = 0.0

    print("=== INDEPENDENT LOCAL TRAINING ===")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Epochs per client: {EPOCHS}")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for client_id, sensor_ids in enumerate(sensor_groups, start=1):
        print(
            f"\nClient {client_id}: "
            f"sensors {sensor_ids[0]}–{sensor_ids[-1]}"
        )

        # Extract only this client's sensor data.
        client_x_train = X_train[:, sensor_ids, :].reshape(
            -1, X_train.shape[-1]
        ).astype(np.float32)

        client_y_train = y_train[:, sensor_ids].reshape(
            -1, 1
        ).astype(np.float32)

        client_x_test = X_test[:, sensor_ids, :].reshape(
            -1, X_test.shape[-1]
        ).astype(np.float32)

        client_y_test = y_test[:, sensor_ids].reshape(
            -1
        ).astype(np.float32)

        train_dataset = TensorDataset(
            torch.from_numpy(client_x_train),
            torch.from_numpy(client_y_train),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        # Each client starts with a completely independent model.
        model = TrafficModel(output_sensors=1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )
        loss_fn = nn.MSELoss()

        start = time.time()

        for epoch in range(EPOCHS):
            loss = train_model(
                model, train_loader, optimizer, loss_fn
            )

        training_seconds = time.time() - start
        total_training_time += training_seconds

        model.eval()
        test_tensor = torch.from_numpy(client_x_test)

        with torch.no_grad():
            predictions = model(test_tensor).numpy().ravel()

        mae = mean_absolute_error(client_y_test, predictions)
        rmse = float(
            np.sqrt(mean_squared_error(client_y_test, predictions))
        )

        print(f"Examples: {len(train_dataset)}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Training time: {training_seconds:.2f}s")

        all_predictions.extend(predictions.tolist())
        all_actual.extend(client_y_test.tolist())

        client_results.append({
            "client": client_id,
            "sensors": [
                int(sensor_ids[0]), int(sensor_ids[-1])
            ],
            "mae": float(mae),
            "rmse": rmse,
            "training_seconds": round(training_seconds, 2),
            "training_examples": len(train_dataset),
        })

        torch.save(
            model.state_dict(),
            output_dir / f"local_client_{client_id}.pt",
        )

    # Aggregate predictions across each client's own test sensors.
    overall_mae = mean_absolute_error(
        all_actual, all_predictions
    )
    overall_rmse = float(
        np.sqrt(mean_squared_error(all_actual, all_predictions))
    )

    results = {
        "model": "Independent Local Models",
        "clients": NUM_CLIENTS,
        "epochs_per_client": EPOCHS,
        "overall_mae": float(overall_mae),
        "overall_rmse": overall_rmse,
        "total_training_seconds": round(total_training_time, 2),
        "client_results": client_results,
        "seed": SEED,
    }

    print("\n=== OVERALL LOCAL-ONLY RESULTS ===")
    print(f"MAE: {overall_mae:.6f}")
    print(f"RMSE: {overall_rmse:.6f}")
    print(f"Total training time: {total_training_time:.2f}s")

    with open(output_dir / "local_only_results.json", "w") as file:
        json.dump(results, file, indent=2)

    pd.DataFrame(client_results).to_csv(
        output_dir / "local_only_clients.csv",
        index=False,
    )

    print("\nSaved local-only results under results/.")


if __name__ == "__main__":
    main()
