# Udacity Banana Collector

This project demonstrates how to train an agent to collect bananas in a room using Deep Q-Networks algorithm.

## Environment

The environment is a modified version of Unity ML-Agents [Banana-Collector][banana-collector].

[banana-collector]: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector

### Goal

The goal of the agent is to collect yellow bananas though avoiding blue bananas. The environment is considered to be solved when the average return for the consecutive 100 episode is over 13.

### Reward

The agent gets +1 reward when it reaches a yellow banana and -1 when it does a blue one.

  * +1 - yellow banana
  * -1 - blue banana

### Observation Space

The observation space has 37 dimensions and contains the agent's velocity plus ray-based perception of objects around the agent's forward direction.

### Action Space

Based on the observation, the agent needs to learn how to best select actions. Four discrete actions are available:

  * 0 - move forward
  * 1 - move backward
  * 2 - turn left
  * 3 - turn right

## Getting Started

### Installing Dependencies

```
$ conda env create -f conda-environment.yml -n banana
$ conda activate banana
(banana) $
```

### Downloading Unity ML-Agents Environment (__optional_)

You can skip this step since Banana_Linux is already included in this git repository.

```
$ curl -LO https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
$ unzip Banana_Linux.zip
```

### Training The Agent

```
(banana) $ python -m train
```

### Running

```
(banana) $ python -m play
```
