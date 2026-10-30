
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat


DATA_PATH = Path("data/raw/traffic_dataset.mat")


def convert_cells(cell_array):
    """Convert MATLAB cells into a NumPy array."""
    samples = []

    for cell in cell_array.ravel():
        if sparse.issparse(cell):
            sample = cell.toarray()
        else:
            sample = np.asarray(cell)

        samples.append(sample)

    return np.stack(samples, axis=0)


def load_dataset(path=DATA_PATH):
    """Load inputs, targets, and sensor connectivity."""
    data = loadmat(path)

    X_train = convert_cells(data["tra_X_tr"])
    X_test = convert_cells(data["tra_X_te"])

    # Convert targets from (sensors, samples) to (samples, sensors).
    y_train = np.asarray(data["tra_Y_tr"]).T
    y_test = np.asarray(data["tra_Y_te"]).T

    adjacency = np.asarray(data["tra_adj_mat"])

    return X_train, X_test, y_train, y_test, adjacency


def main():
    X_train, X_test, y_train, y_test, adjacency = load_dataset()

    arrays = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "adjacency": adjacency,
    }

    print("=== LOADED DATASET ===")
    for name, array in arrays.items():
        print(
            f"{name}: shape={array.shape}, "
            f"dtype={array.dtype}, "
            f"finite={np.isfinite(array).all()}"
        )

    print("\n=== VALUE RANGES ===")
    for name, array in arrays.items():
        print(f"{name}: min={array.min():.6f}, max={array.max():.6f}")

    assert X_train.shape == (1261, 36, 48)
    assert X_test.shape == (840, 36, 48)
    assert y_train.shape == (1261, 36)
    assert y_test.shape == (840, 36)
    assert adjacency.shape == (36, 36)

    assert all(np.isfinite(array).all() for array in arrays.values())

    print("\nAll shape and finite-value checks passed.")


if __name__ == "__main__":
    main()
