import torch
from agent import *

tensor = torch.tensor

sample = (tensor([[11., 12.],
                  [21., 22.]]),
          tensor([[0.],
                  [2.]]),
          tensor([[1.],
                  [2.]]),
          tensor([[31., 32.],
                  [41., 42.]]),
          tensor([[0],
                  [1]], dtype=torch.uint8))

def test_learn():
    s0, a, r, s1, d = (x[0] for x in sample)
    agent0 = DQNAgent(2, 2, batch_size=1)
    agent0.observe_and_learn(s0, a, r, s1, d)
