import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Q-network used to calculate the values of all actions at a given state"""

    def __init__(self, state_size, n_actions, hidden_units, device='cpu'):
        """Initialize the Q-network"""
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.hidden_unites = hidden_units
        self.device = device

        self.layers = []
        input_size = state_size
        if hidden_units:
            for i, units in enumerate(hidden_units):
                # Submodules have to be registered explicitly since they are not
                # imediate attributes of this module
                layer = nn.Linear(input_size, units)
                self.add_module(f'hidden_{i}', layer)

                self.layers.append(layer)
                input_size = units
        output_layer = nn.Linear(input_size, n_actions)
        self.add_module('out', output_layer)
        self.layers.append(output_layer)

        # TODO when to initialize the network params?

    def forward(self, state):
        """Forwards the state through the network and gets the values for all
        actions
        """
        x = state
        for layer in self.layers[:-1]:
            z = layer(x)
            a = F.relu(z)
            x = a
        return self.layers[-1](x)
