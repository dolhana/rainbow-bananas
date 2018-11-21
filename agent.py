import random
import copy
from collections import namedtuple, deque

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import model


class DQNAgent():
    """DQN agent is an agent that uses DQN as a value network.
    """

    def __init__(self,
                 state_size, n_actions,
                 hidden_units=[64, 64],
                 learning_rate=5e-4,
                 batch_size=64,
                 gamma=0.99,
                 replay_buffer_capacity=10**5,
                 update_interval=4,
                 tau=1e-3,
                 device="cpu"):
        """Initialize the agent

        Args:
          state_size (int): the dimension of a state
          n_actions (int): the number of actions that the agent can choose from
          seed (int): random seed
        """
        self.state_size = state_size
        self.n_actions = n_actions
        self.device = device

        # Hyper-parameters
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_buffer_capacity = replay_buffer_capacity
        self.update_interval = update_interval
        self.tau = tau

        # Deep Q-network
        self.online_network = model.QNetwork(state_size, n_actions, hidden_units).to(device)
        self.target_network = copy.deepcopy(self.online_network)
        self.optimizer = optim.Adam(self.online_network.parameters(), learning_rate)
        self.losser = torch.nn.MSELoss()

        # TODO Does replay_buffer need more ecapsulation?
        self.replay_buffer = []

        self.step_count = 0

    def choose_action(self, state, training=True):
        """The agent chooses an action based on the given state

        Returns:
          int: the action index that maximize the return at a given state
        """
        if random.random() >= self._epsilon():
            return np.argmax(self.action_values(state))
        else:
            return random.randint(0, self.n_actions - 1)

    def _epsilon(self):
        return max(1. - .9 * self.step_count / 1000., 0.1)

    def action_values(self, state):
        """Calculates vales of all actions at a given state

        Returns:
          numpy.array(): the values of all actions
        """
        return self.online_network(state).cpu().detach().numpy()

    def observe_and_learn(self, state, action, reward, next_state, done):
        """Observes and stores an experience
        """
        # step_count is needed for soft-update, (TODO) epsilon shaping and (TODO) learning_rate shaping
        self.step_count += 1

        # TODO Is limiting the size of replay_buffer by slicing efficient enough?
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.replay_buffer = self.replay_buffer[-self.replay_buffer_capacity:]

        # TODO Sample a batch from the replay_buffer
        experience_sample = random.choices(self.replay_buffer, k=self.batch_size)

        self.learn(experience_sample)

        if (self.step_count % self.update_interval == 0
            and len(self.replay_buffer) >= self.batch_size):
            self.soft_update()

    def learn(self, experiences):
        """Learns from experiences
        """
        s0, r, a, s1, done = (np.vstack(e) for e in zip(*experiences))

        # TD target
        s1_value = self.target_network(s1).detach().max(dim=1)
        td_target = r + np.logical_not(done) * self.gamma * s1_value
        expected = self.online_network(s0).gather(dim=1, index=a)

        # Loss
        loss = self.losser.loss(expected, td_target)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backword()
        self.optimizer.step()

    def soft_update(self):
        """Updates the target network so that it moves toward to the online network
        """
        for target_param, online_param in \
                zip(self.target_network.parameters(), self.online_network.parameters()):
            target_param.data.copy_(self.tau * target_param + (1. - self.tau) * online_param)


def manual_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
