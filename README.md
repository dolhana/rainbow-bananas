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

```bash
conda env create -f conda-environment.yml -n banana
conda activate banana
```

### Downloading Unity ML-Agents Environment (__optional_)

You can skip this step since Banana_Linux is already included in this git repository.

```bash
curl -LO https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
```

This project assumes Linux but the binaries for other OSes are also available to download.

* [Banana_Linux.zip for Linux][banana-linux-zip]
* [Banana.app.app for Mac OSX][banana-osx-zip]
* [Banana_Windows_x86.zip for Windows (32-bit)][banana-windows-x86]
* [Banana_Windows_x86_64.zip for Windows (64-bit)][banana-windows-x86-64]

[banana-linux-zip]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
[banana-osx-zip]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
[banana-windows-x86]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
[banana-windows-x86-64]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

### Training The Agent

```bash
(banana) $ python -m train
```

### Running

```bash
(banana) $ python -m play
```
