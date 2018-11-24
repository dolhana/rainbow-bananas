import torch
from replaybuffer import ReplayBuffer


def test_store_and_sample_small():
    replay = ReplayBuffer(5, 2)

    # passing a list
    replay.store([0, 1], 2, 3, [1, 2], False)

    assert (replay._insert == 1)
    assert (replay._full == False)

    for i in range(1, 6):
        replay.store([i, i+1], i+2, i+3, [i+1, i+2], False)

    assert (replay._insert == 1)
    assert (replay._full)

    states, actions, rewards, next_states, dones = (
        replay.sample(2))

    assert (states.size() == torch.Size([2, 2]))
    assert (actions.size() == torch.Size([2]))
    assert (rewards.size() == torch.Size([2]))
    assert (next_states.size() == torch.Size([2, 2]))
