
import numpy as np

from load_dataset import load_dataset


def describe(name, array):
    print(f"\n=== {name} ===")
    print("Shape:", array.shape)
    print("Minimum:", float(np.min(array)))
    print("Maximum:", float(np.max(array)))
    print("Mean:", float(np.mean(array)))
    print("Standard deviation:", float(np.std(array)))
    print("Unique values (up to 15):", np.unique(array)[:15])


def main():
    X_train, X_test, y_train, y_test, adjacency = load_dataset()

    print("=== INPUT FEATURE RANGES ===")
    print("Feature | Min | Max | Mean | Std")

    # Samples x sensors x features
    for feature in range(X_train.shape[2]):
        values = X_train[:, :, feature]

        print(
            f"{feature:>7} | "
            f"{values.min():.5f} | "
            f"{values.max():.5f} | "
            f"{values.mean():.5f} | "
            f"{values.std():.5f}"
        )

    describe("Training targets", y_train)
    describe("Test targets", y_test)

    print("\n=== SAMPLE INPUT ===")
    print("First sample, first sensor:")
    print(X_train[0, 0, :])

    print("\n=== SAMPLE TARGETS ===")
    print("First sample, all 36 sensors:")
    print(y_train[0])

    print("\n=== TEMPORAL CHECK ===")
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("First 5 training targets for sensor 0:", y_train[:5, 0])
    print("Last 5 training targets for sensor 0:", y_train[-5:, 0])
    print("First 5 test targets for sensor 0:", y_test[:5, 0])


if __name__ == "__main__":
    main()
