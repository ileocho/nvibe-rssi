"""This module contains all utility functions used for training and evaluating the model."""
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: nn.Module,
    loader: torch.utils.data.DataLoader,
):
    """A training loop to fit the model to the data.

    Args:
        model: the neural network model to train.
        criterion: the loss function.
        optimizer: the optimizer used to update the model parameters.
        loader: the data loader to iterate over the training data.

    Returns:
        The average loss over the training data for the current epoch.
    """
    model.train()
    train_loss = 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(loader)


def validation_loop(
    model: nn.Module, criterion: nn.Module, loader: torch.utils.data.DataLoader
):
    """A validation loop to evaluate the model on the validation data. This function does not update the model parameters.

    Args:
        model: a trained neural network model.
        criterion: the loss function.
        loader: the data loader to iterate over the validation data.

    Returns:
        The average loss over the validation data for the current epoch.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(loader)


def fit_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    n_epochs: int,
):
    """A function to train and evaluate a model.

    Args:
        model: the neural network model to train and evaluate.
        criterion: the loss function.
        optimizer: the optimizer used to update the model parameters.
        train_loader: the data loader to iterate over the training data.
        test_loader: the data loader to iterate over the test data.
        n_epochs: the number of epochs to train the model for.
    """
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        train_loss = train_loop(model, criterion, optimizer, train_loader)
        val_loss = validation_loop(model, criterion, test_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
        )
    plot_losses(train_losses, val_losses)


def plot_losses(train_losses, val_losses):
    """Plot the training and validation losses using Plotly.

    Args:
        train_losses: the training losses, a list of shape (n_epochs,).
        val_losses: the validation losses, a list of shape (n_epochs,).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(y=train_losses, mode="lines", name="Training Loss"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(y=val_losses, mode="lines", name="Validation Loss"),
        secondary_y=False,
    )
    fig.update_layout(title="Losses", xaxis_title="Epoch", yaxis_title="Loss")
    fig.update_layout(width=600 * 1.34, height=600)
    fig.show()
