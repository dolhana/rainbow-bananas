import torch
from replaybuffer import ReplayBuffer


def test_store_and_sample_basic():
    replay = ReplayBuffer(5, 2)

    # passing a list
    replay.store([0, 1], 2, 3, [1, 2], False)

    # if not enough stored, returns empty tensors
    assert all(len(x) == 0 for x in replay.sample(10))

    assert (replay._insert == 1)
    assert (replay._full == False)
    rewards = replay._rewards[:replay._insert]
    assert (torch.equal(rewards, torch.tensor([[3]], dtype=torch.float)))

def test_sample_shapes():
    replay = ReplayBuffer(5, 2)

    # passing a list
    replay.store([0, 1], 2, 3, [1, 2], False)

    for i in range(1, 6):
        replay.store([i, i+1], i+2, i+3, [i+1, i+2], False)

    assert (replay._insert == 1)
    assert (replay._full)

    states, actions, rewards, next_states, dones = (
        replay.sample(2))

    assert (states.size() == torch.Size([2, 2]))
    assert (actions.size() == torch.Size([2, 1]))
    assert (rewards.size() == torch.Size([2, 1]))
    assert (next_states.size() == torch.Size([2, 2]))
    assert (dones.size() == torch.Size([2, 1]))

def test_sample_randomness():
    replay = ReplayBuffer(2, 1)
    replay.store(1, 1, 1, 1, False)
    replay.store(2, 2, 2, 2, True)

    states = replay.sample(2)[0]
    assert not torch.equal(states[0], states[1])
