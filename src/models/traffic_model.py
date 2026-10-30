
import torch
from torch import nn


class TrafficModel(nn.Module):
    def __init__(self, input_features=48, hidden_size=64, output_sensors=36):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_sensors),
        )

    def forward(self, x):
        return self.network(x)
