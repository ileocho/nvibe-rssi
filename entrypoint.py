import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from pathlib import Path
import time

from model.model import NeuralNet
from model.train_utils import fit_model
from data.data_utils import load_data, scale_data, apply_pca, get_dataloaders, create_output_folder
from visualization.results_visualisation import plot_error_hist, plot_predictions, plot_ecdf

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data-path", "-data", type=str, required=True, help="Path to the data folder containing train and test folders.")
parser.add_argument("--batch-size", "-bs", type=int, default=16, help="Batch size for the data loaders.")
parser.add_argument("--epochs", "-e",type=int, default=100, help="Number of epochs for training.")
parser.add_argument("--hidden-size", type=int, default=128, help="Number of neurons in the hidden layer.")
parser.add_argument("--components", type=int, default=64, help="Number of output features after PCA.")
parser.add_argument("--output", "-o", required=False, default="output", help="Path to the output folder.")
parser.add_argument("--plot", required=False, action="store_true", help="Plot the results.")

if __name__ == "__main__":
    args = parser.parse_args()

    # Loading the data into numpy arrays
    train_features, train_targets, test_features, test_targets = load_data(Path(args.data_path))

    # Scaling the features and targets accordingly
    scaler = StandardScaler()
    train_features_scaled, test_features_scaled = scale_data(
        scaler, train_features, test_features
    )
    train_targets_scaled, test_targets_scaled = scale_data(
        scaler, train_targets, test_targets
    )

    # Applying PCA to the features
    train_features_pca, test_features_pca = apply_pca(
        train_features_scaled, test_features_scaled, n_components=args.components
    )

    n_samples = train_features.shape[0]

    # Converting numpy arrays to PyTorch tensors
    train_features_tensor = torch.tensor(train_features_pca, dtype=torch.float32)
    test_features_tensor = torch.tensor(test_features_pca, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets_scaled, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets_scaled, dtype=torch.float32)

    # Creating DataLoaders
    batch_size = args.batch_size
    train_loader, test_loader = get_dataloaders(
        train_features_tensor,
        train_targets_tensor,
        test_features_tensor,
        test_targets_tensor,
        batch_size,
    )

    # Define the model, criterion, and optimizer
    input_size = train_features_tensor.shape[1]
    hidden_size = args.hidden_size
    output_size = train_targets_tensor.shape[1]

    model = NeuralNet(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.Adam(model.parameters())  # Adam optimizer

    # Training and validation
    n_epochs = args.epochs
    fit_model(model, criterion, optimizer, train_loader, test_loader, n_epochs)

    # Make predictions
    model.eval()
    predictions = model(test_features_tensor).detach().numpy()

    # Invert the predictions and the targets to their original scales
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(test_targets_tensor)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse}")

    # create output folder that contains name of the data folder + timestamp
    output_path = Path(args.output) / f"{Path(args.data_path).name}_{int(time.time())}"
    print(output_path)
    if args.plot:
        if not output_path.exists():
            output_path.mkdir(parents=True)
        # Plot the predictions
        plot_predictions(y_test, predictions, output_path)
        plot_error_hist(y_test, predictions, output_path)
        plot_ecdf(y_test, predictions, output_path)
