import torch


class ReplayBuffer:

    def __init__(self, capacity, state_size, device='cpu'):
        self.capacity = capacity
        self.state_size = state_size

        self._insert = 0
        self._full = False
        self._states = torch.empty(capacity, state_size, dtype=torch.float)
        self._actions = torch.empty(capacity, dtype=torch.uint8)
        self._rewards = torch.empty(capacity, dtype=torch.float)
        self._next_states = torch.empty(capacity, state_size, dtype=torch.float)
        self._dones = torch.empty(capacity, dtype=torch.uint8)

    def store(self, state, action, reward, next_state, done):
        self._states[self._insert] = torch.as_tensor(state, dtype=torch.float)
        self._actions[self._insert] = torch.as_tensor(action, dtype=torch.float)
        self._rewards[self._insert] = torch.as_tensor(reward, dtype=torch.float)
        self._dones[self._insert] = torch.as_tensor(done, dtype=torch.uint8)
        self._insert += 1
        if self._insert == self.capacity:
            self._full = True
            self._insert = 0

    def sample(self, size):
        end = self._insert if not self._full else self.capacity
        idx = torch.randint(low=0, high=end, size=(size,), dtype=torch.int64)
        states = torch.index_select(self._states, 0, idx)
        actions = torch.index_select(self._actions, 0, idx)
        rewards = torch.index_select(self._rewards, 0, idx)
        next_states = torch.index_select(self._next_states, 0, idx)
        dones = torch.index_select(self._dones, 0, idx)
        return states, actions, rewards, next_states, dones
