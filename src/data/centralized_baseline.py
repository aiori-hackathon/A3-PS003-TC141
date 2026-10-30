
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from load_dataset import load_dataset


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")
    start = time.time()

    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    print("Generating test predictions...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    print(f"{name} MAE:  {mae:.6f}")
    print(f"{name} RMSE: {rmse:.6f}")
    print(f"Training time: {train_seconds:.2f} seconds")

    return {
        "model": name,
        "mae": float(mae),
        "rmse": rmse,
        "training_seconds": round(train_seconds, 2),
    }


def main():
    print("Loading traffic dataset...")
    X_train, X_test, y_train, y_test, adjacency = load_dataset()

    # Each sensor at each time step becomes one supervised example.
    # The corresponding target is that sensor's traffic volume.
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_test = X_test.reshape(-1, X_test.shape[-1])
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    print("\n=== TRAINING DATA ===")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    models = [
        (
            "Mean baseline",
            DummyRegressor(strategy="mean"),
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = []

    for name, model in models:
        results.append(
            evaluate_model(
                name, model, X_train, y_train, X_test, y_test
            )
        )

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "centralized_baseline.csv", index=False)

    with open(output_dir / "centralized_baseline.json", "w") as file:
        json.dump(results, file, indent=2)

    print("\n=== BASELINE COMPARISON ===")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
