import torch


class ReplayBuffer:

    def __init__(self, capacity, state_size, device='cpu'):
        self.capacity = capacity
        self.state_size = state_size

        self._insert = 0
        self._full = False
        self._states = torch.empty(capacity, state_size, dtype=torch.float)
        self._actions = torch.empty(capacity, 1, dtype=torch.uint8)
        self._rewards = torch.empty(capacity, 1, dtype=torch.float)
        self._next_states = torch.empty(capacity, state_size, dtype=torch.float)
        self._dones = torch.empty(capacity, 1, dtype=torch.uint8)

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
        """ Returns a sample with `size` elements

        Returns:
          (states, actions, rewards, next_states, dones):
            if # of stored experiences >= size
          (tensor([]), tensor([]), tensor([]), tensor([]), tensor([])):
            otherwise
        """
        end = self._insert if not self._full else self.capacity
        # If no stored experience, return empty tensors
        if end == 0 or end < size:
            empty_tensor = torch.empty(size=(0,))
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor
        idx = torch.randperm(end)[:size]
        states = torch.index_select(self._states, 0, idx)
        actions = torch.index_select(self._actions, 0, idx)
        rewards = torch.index_select(self._rewards, 0, idx)
        next_states = torch.index_select(self._next_states, 0, idx)
        dones = torch.index_select(self._dones, 0, idx)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._insert if not self._full else self.capacity
