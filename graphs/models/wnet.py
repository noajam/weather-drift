"""
WNet model class
"""
import torch
import torch.nn as nn

from graphs.weights_initializer import init_model_weights

class WNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # define layers
        self.relu = nn.ReLU()
        
        self.linearIn = nn.Linear(in_features=self.config.input_channels, out_features=self.config.hidden_dim)
        self.linearOut = nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.num_classes)
        
        self.softmax = nn.Softmax(dim=1)

        # initialize weights
        self.apply(init_model_weights)

    def forward(self, x):
        print(x)
        x = x.float()
        x = self.linearIn(x)
        x = self.relu(x)
        x = self.linearOut(x)
        x = self.relu(x)
        x = self.softmax(x)

        return x
