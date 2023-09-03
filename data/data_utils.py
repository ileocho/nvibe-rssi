import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def lon_lat_to_meters(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Calculates the distance between two points in (latitute, longitude) format to meters.

    Args:
        lat1 (float): Latitude of first point.
        long1 (float): Longitude of first point.
        lat2 (float): Latitude of second point.
        long2 (float): Longitude of second point.

    Returns:
        distance (float): Distance between two points in meters.
    """

    # approximate radius of earth in km
    R = 6373.0

    # convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(long1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(long2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # convert to meters
    distance = R * c * 1000

    return distance


def ecdf_plot(data: list, label: str) -> None:
    """Plots the ECDF of a data array.

    Args:
        data: Array of data to plot.
        label: Label for the plot.
    """
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    # Plot the ECDF
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, marker='.', linestyle='none', label=label)
    plt.xlabel('measurement')
    plt.ylabel('ECDF')
    plt.legend()
    plt.show()

def load_data(data_folder: Path):
    """Load the data from the data folder by searching for nested train/test folders.

    Args:
        data_folder: The path to the data folder.

    Returns:
        x_train: The training features.
        y_train: The training targets.
        x_test: The test features.
        y_test: The test targets.
    """

    # Find the train and test folders
    train_folder = data_folder / "train"
    test_folder = data_folder / "test"

    # Find the train and test files
    x_train_file = train_folder / "x_train.csv"
    y_train_file = train_folder / "y_train.csv"
    x_test_file = test_folder / "x_test.csv"
    y_test_file = test_folder / "y_test.csv"

    # Load the data and drop the first column (index)
    x_train = np.loadtxt(x_train_file, delimiter=",", skiprows=1)
    x_train = np.delete(x_train, 0, axis=1)
    y_train = np.loadtxt(y_train_file, delimiter=",", skiprows=1)
    y_train = np.delete(y_train, 0, axis=1)

    x_test = np.loadtxt(x_test_file, delimiter=",", skiprows=1)
    x_test = np.delete(x_test, 0, axis=1)
    y_test = np.loadtxt(y_test_file, delimiter=",", skiprows=1)
    y_test = np.delete(y_test, 0, axis=1)

    print(f"Train features shape: {x_train.shape}")
    print(f"Train targets shape: {y_train.shape}")
    print(f"Test features shape: {x_test.shape}")
    print(f"Test targets shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test


def scale_data(scaler: StandardScaler, train_data: np.ndarray, test_data: np.ndarray):
    """Scale the train and test features to have zero mean and unit variance.

    Args:
        scaler: The scaler used to scale the data.
        train_features: The train features.
        test_features: The test features.

    Returns:
        train_features_scaled: The scaled train features.
        test_features_scaled: The scaled test features.
    """
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    return scaled_train_data, scaled_test_data


def apply_pca(train_features: np.ndarray, test_features: np.ndarray, n_components: int):
    """Applies PCA to the train and test features.

    Args:
        train_features: the train dataset.
        test_features: the test dataset.
        n_components: the number of output features after PCA.

    Returns:
        train_features_pca: the train dataset after PCA transformation
        test_features_pca: the test dataset after PCA transformation
    """
    pca = PCA(n_components=n_components)
    train_features_pca = pca.fit_transform(train_features)
    test_features_pca = pca.transform(test_features)

    return train_features_pca, test_features_pca


def get_dataloaders(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    batch_size: int,
):
    """Create train and test PyTorch DataLoaders.

    Args:
        train_features: the train dataset.
        train_targets: the train targets.
        test_features: the test dataset.
        test_targets: the test targets.
        batch_size: the batch size for the DataLoaders.

    Returns:
        train_loader: the PyTorch DataLoader for the train dataset.
        test_loader: the PyTorch DataLoader for the test dataset.
    """
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(train_features), torch.tensor(train_targets)
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(test_features), torch.tensor(test_targets)
        ),
        batch_size=batch_size,
    )

    return train_loader, test_loader


def create_output_folder(output: Path, data_folder: Path):
    """This function formats the output folder to ensure correct tracking of the results.

    Args:
        output: the path to the output folder
        data_folder: the path to the datafolder used for this experiment

    Raises:
        TypeError: 
    """
    # Check if output is a valid Path object
    if not isinstance(output, Path):
        raise TypeError("output must be a Path object")

    # Check if data_folder is a valid Path object
    if not isinstance(data_folder, Path):
        raise TypeError("data_folder must be a Path object")

    # Check if data_folder exists
    if not data_folder.exists():
        raise ValueError(f"data_folder {data_folder} does not exist")
    
    # Merge output and data_folder
    output_folder = output

    # Create output directory if it does not exist
    if not output.exists():
        output.mkdir(parents=True)
        print(f"Created output directory {output}")

    # If output directory already exists
    else:
        print(f"Output directory {output} already exists")

    return output