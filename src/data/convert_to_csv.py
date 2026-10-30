
from pathlib import Path

import pandas as pd

from load_dataset import load_dataset


OUTPUT_DIR = Path("data/processed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, adjacency = load_dataset()

    def save_inputs(array, filename):
        samples, sensors, features = array.shape

        rows = array.reshape(samples * sensors, features)

        columns = [f"feature_{i}" for i in range(features)]
        df = pd.DataFrame(rows, columns=columns)

        df.insert(
            0,
            "sensor_id",
            list(range(sensors)) * samples,
        )
        df.insert(
            0,
            "sample_index",
            [i for i in range(samples) for _ in range(sensors)],
        )

        df.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"Saved {filename}: {df.shape}")

    def save_targets(array, filename):
        columns = [f"sensor_{i}" for i in range(array.shape[1])]
        df = pd.DataFrame(array, columns=columns)
        df.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"Saved {filename}: {df.shape}")

    save_inputs(X_train, "X_train.csv")
    save_inputs(X_test, "X_test.csv")

    save_targets(y_train, "y_train.csv")
    save_targets(y_test, "y_test.csv")

    adjacency_df = pd.DataFrame(
        adjacency,
        index=[f"sensor_{i}" for i in range(36)],
        columns=[f"sensor_{i}" for i in range(36)],
    )
    adjacency_df.to_csv(OUTPUT_DIR / "adjacency.csv")
    print(f"Saved adjacency.csv: {adjacency_df.shape}")

    print(f"\nCSV files created in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
