import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize the agent

        Args:
          state_size (int): dimension of each state
          action_size (int): dimension of each action
          seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)

    def step(self):
        """The agent takes one step in the environment"""

    def choose_action(self, state, epsilon=0.):
        """The agent chooses an action based on the given state"""

    def learn(self, experiences, gamma):
        """The agent learns from the experience"""

    def soft_update(self, tau):
        """Updates the target network so that it moves toward to the local network"""
