
from ucimlrepo import fetch_ucirepo


def main():
    print("Fetching UCI Traffic Flow Forecasting dataset...")

    dataset = fetch_ucirepo(id=734)

    X = dataset.data.features
    y = dataset.data.targets

    print("\n=== DATASET METADATA ===")
    print("Dataset name:", dataset.metadata.get("name"))
    print("Dataset ID:", dataset.metadata.get("uci_id"))
    print("License:", dataset.metadata.get("license"))

    print("\n=== FEATURES ===")
    print("Shape:", X.shape)
    print("Number of rows:", len(X))
    print("Number of features:", X.shape[1])
    print("Feature names:")
    print(X.columns.tolist())

    print("\n=== TARGET ===")
    print("Shape:", y.shape)
    print("Target names:", y.columns.tolist())

    print("\n=== FIRST FIVE FEATURE ROWS ===")
    print(X.head())

    print("\n=== FIRST FIVE TARGET ROWS ===")
    print(y.head())

    print("\n=== MISSING VALUES ===")
    print("Missing feature values:", int(X.isna().sum().sum()))
    print("Missing target values:", int(y.isna().sum().sum()))

    print("\n=== DATA TYPES ===")
    print(X.dtypes.value_counts())

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
