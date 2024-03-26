import numpy as np
import pandas as pd
import torch
from typing import *

from torch import nn
from get_data import *


class ANN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_layers[0]), nn.LeakyReLU())
        self.layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.LeakyReLU()) for i in
              range(1, len(hidden_layers))])
        self.last_layer = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        return self.last_layer(self.layers(self.first_layer(x)))
