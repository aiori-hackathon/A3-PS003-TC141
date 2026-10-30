
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.load_dataset import load_dataset
from src.models.traffic_model import TrafficModel


SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001


def prepare_data(X, y):
    # Each sensor at each time sample becomes one training example.
    X = X.reshape(-1, X.shape[-1]).astype(np.float32)
    y = y.reshape(-1).astype(np.float32)

    return (
        torch.from_numpy(X),
        torch.from_numpy(y.reshape(-1, 1)),
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, _ = load_dataset()

    X_train_tensor, y_train_tensor = prepare_data(X_train, y_train)
    X_test_tensor, y_test_tensor = prepare_data(X_test, y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TrafficModel(output_sensors=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("Training centralized neural network...")
    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(inputs)

        running_loss /= len(train_loader.dataset)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} "
                f"training MSE: {running_loss:.6f}"
            )

    training_seconds = time.time() - start

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().ravel()

    actual = y_test_tensor.numpy().ravel()

    mae = mean_absolute_error(actual, predictions)
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

    print("\n=== CENTRALIZED NEURAL NETWORK ===")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Training time: {training_seconds:.2f} seconds")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "Centralized Neural Network",
        "mae": float(mae),
        "rmse": rmse,
        "training_seconds": round(training_seconds, 2),
        "epochs": EPOCHS,
        "seed": SEED,
    }

    with open(output_dir / "centralized_neural_network.json", "w") as file:
        json.dump(results, file, indent=2)

    torch.save(model.state_dict(), output_dir / "centralized_neural_network.pt")
    print("Saved metrics and model under results/.")


if __name__ == "__main__":
    main()
