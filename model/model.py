""" This module contains the definition of the neural network used for the prediction of a longitude and latitude based on RSSI values. """
import torch
import torch.nn as nn


# Define the Neural Network class
class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """A simple neural network with 3 layers.

        Args:
            input_size: The input size of the network, represents the number of input features.
            hidden_size: The number of neurons in the hidden layer.
            output_size: The output size of the network, here, the longitute and latitude of the location to predict.
        """
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x: The input data, a tensor of shape (batch_size, input_size).

        Returns:
            The output of the network, a tensor of shape (batch_size, output_size).
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x =  self.dropout(x)
        x = self.layer3(x)
        return x
