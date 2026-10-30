
from scipy.io import loadmat
from scipy import sparse
import numpy as np


def show_values(item, count=10):
    if sparse.issparse(item):
        return item.toarray().ravel()[:count]
    return np.asarray(item).ravel()[:count]


def main():
    data = loadmat("data/raw/traffic_dataset.mat")

    for name in ["tra_X_tr", "tra_X_te"]:
        cells = data[name]
        print(f"\n=== {name} ===")
        print("Container shape:", cells.shape)
        print("Container dtype:", cells.dtype)

        for i in range(min(3, cells.size)):
            item = cells.flat[i]
            print(f"Cell {i}: shape={item.shape}, dtype={item.dtype}")
            print("Sample values:", show_values(item))

    for name in ["tra_Y_tr", "tra_Y_te"]:
        y = data[name]
        print(f"\n=== {name} ===")
        print("Shape:", y.shape)
        print("First 5 values of sensor 0:", y[0, :5])
        print("Minimum:", np.nanmin(y))
        print("Maximum:", np.nanmax(y))
        print("NaN count:", np.isnan(y).sum())

    adj = data["tra_adj_mat"]
    print("\n=== tra_adj_mat ===")
    print("Shape:", adj.shape)
    print("First 5 rows and columns:")
    print(adj[:5, :5])
    print("Minimum:", np.nanmin(adj))
    print("Maximum:", np.nanmax(adj))
    print("Nonzero entries:", np.count_nonzero(adj))

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
