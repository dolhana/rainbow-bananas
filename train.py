"""Trains the agent using Deep Q-Learning"""
import random
from collections import deque
import tqdm
import numpy as np
import torch
from unityagents import UnityEnvironment
from dqn_agent import Agent


def dqn(env, agent, n_episodes=1000, max_t=1000, eps_start=1., eps_end=0.01,
        eps_decay=0.995, show_progress=True):
    """Deep Q-Learning

    Args:
      n_episodes (int): maximum number of training episodes
      max_t (int): maximum number of timesteps per episodes
      eps_start (float): starting value of epsilon, for epsilon-greedy action selection
      eps_end (float): minimum value of epsilon
      eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # Get the default brain
    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    progress = tqdm.tqdm(range(1, n_episodes + 1), disable=not show_progress)
    for i_episode in progress:
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for _ in range(max_t):
            action = agent.act(state)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        scores_mean = np.mean(scores_window)
        #print(f'\rEpisode {i_episode}\tAverage Score: {scores_mean:.2f}', end='')
        progress.set_postfix({'episode': i_episode, 'score_mean': scores_mean})

        if scores_mean >= 13.:
            print(f'\nEnvironment solved in {i_episode - 100} epsidoes!')
            print(f'Average Score: {scores_mean}')
            torch.save(agent.qnetwork_local.state_dict(), 'qnetwork_local_checkpoint.pth')
            break

        eps = max(eps_end, eps_decay * eps)

    return scores


def train(env, n_episodes=1000, max_t=1000, agent_kwargs={}, seed=None,
          save_filename='p1_dqn_agent.pth', show_progress=True, **kwargs):
    """Trains the agent
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    # The number of actions
    action_size = brain.vector_action_space_size

    # The dimension of state vector
    state_size = len(env_info.vector_observations[0])

    # Create an agent
    agent_kwargs = agent_kwargs.copy()
    agent_kwargs.update({
        'state_size': state_size,
        'action_size': action_size
    })
    agent = Agent(**agent_kwargs)

    # Run DQN algorithm
    scores = dqn(env, agent, n_episodes, max_t, show_progress, **kwargs)

    # Save the trained parameters
    if save_filename is not None:
        torch.save(agent.qnetwork_local.state_dict(), save_filename)

    return agent, scores


# if __name__ == '__main__':
#     env = UnityEnvironment(file_name='Banana_Linux_NoVis/Banana.x86_64')
#     train(env)
