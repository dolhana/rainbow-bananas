import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from replaybuffer import ReplayBuffer


# replay buffer size
BUFFER_SIZE = int(1e5)

# minibatch size
BATCH_SIZE = 64

# discount factor
GAMMA = 0.99

# for soft update of target parameters
TAU = 1e-3

# learning rate
LR = 5e-4

# how often to update the network
UPDATE_EVERY = 4


class Agent():
    """Interacts with and learns from the environment"""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128, device=torch.device('cpu')):
        """DQN agent

        Args:
          state_size (int): dimension of each state
          action_size (int): dimension of each action (or the number of action choices)
          seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size,
                                       fc1_units=fc1_units, fc2_units=fc2_units).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size,
                                        fc1_units=fc1_units, fc2_units=fc2_units).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Initialze qnetwork_target parameters to qnetwork_local
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device=self.device)

        # Initialize the time step counter (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subnet and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Args:
          state (array_like): current state
          eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Set qnetwork_local to evaluation mode
        self.qnetwork_local.eval()

        # This operation should not be included in gradient calculation
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Set back qnetwork_local to training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Args:
          experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
          gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q tagets for current states with actual rewards
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ----- Update the target network -----
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        theta_target = tau * theta_local + (1 - tau) * theta_target

        Args:
          local_model (torch.nn.Module): weights will be copied from
          target_model (torch.nn.MOdule): weights will be copied to
          tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1. - tau) * target_param.data)
