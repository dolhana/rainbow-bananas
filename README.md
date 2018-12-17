# Udacity Banana Collector

This project demonstrates how to train an agent to collect bananas in a room using Deep Q-Learning algorithm.

## Environment

The environment is a modified version of Unity ML-Agents [Banana-Collector][banana-collector].

[banana-collector]: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector

### Goal

The goal of the agent is to collect yellow bananas while avoiding blue bananas. The environment is considered to be solved when the average return for the consecutive 100 episode is greater than or equal to 13.

### Reward

The agent gets +1 reward when it reaches a yellow banana and -1 when it does a blue one.

* +1 - yellow banana
* -1 - blue banana

### Observation Space

The observation space has 37 dimensions and contains the agent's velocity plus ray-based perception of objects around the agent's forward direction.

### Action Space

Based on the observation, the agent needs to learn how to select the best actions. Four discrete actions are available:

* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right

## Getting Started

### Preparation

You need to install the required python packages to be able to run this project. The required python packages are listed in `environment.yml`, which is a [conda](https://conda.io/docs/index.html) environment file. It will create a conda environment named `drlnd`.


```bash
conda env create -f conda-environment.yml
conda activate drlnd
```

Next, you need to download one of the binaries that actually provide the environment that you will run you agent in. Choose the one that matches your operating system.

* [Banana_Linux.zip for Linux][banana-linux]
* [Banana_Linux_NoVis.zip for Linux (headless)][banana-linux-headless]
* [Banana.app.aip for Mac OSX][banana-osx]
* [Banana_Windows_x86.zip for Windows (32-bit)][banana-windows-x86]
* [Banana_Windows_x86_64.zip for Windows (64-bit)][banana-windows-x86-64]

Once you downloaded the file, unzip it at the root of this project. For example, if you run this project in Linux without access to the display,

```bash
$ cd /path/to/udacity-drlnd-p1-navigation
$ curl -LO https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
$ unzip Banana_Linux_NoVis.zip
$ ls -l Banana_Linux_NoVis
total 59324
drwxr-xr-x 7 handol handol     4096 Jun 26 20:34 Banana_Data
-rwxr-xr-x 1 handol handol 30358716 Jan 29  2018 Banana.x86
-rwxr-xr-x 1 handol handol 30382872 Jan 29  2018 Banana.x86_64
```

Now, you're ready to train the agent.

[banana-linux]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
[banana-linux-headless]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
[banana-osx]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
[banana-windows-x86]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
[banana-windows-x86-64]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

## Training The Agent

You can see how to train the agent in Jupyter notebook `Report.ipnb`. To run the notebook, first start up Jupyter server with the following command.

```bash
(base)$ conda activate drlnd
(drlnd)$ jupyter notebook
[I 07:12:35.811 NotebookApp] Serving notebooks from local directory: /home/handol/Code
...
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=e406adf4c029503d5eae4888ec3bc9eb8b7f9a653e9d06bd
```

It will open up the web browser. If not, open the browser manually and go to the URL as noted in the output.

From the file list, open up `Report.ipnb` notebook. If you runs the notebook, the agent successfully achieves the goal within 1000 episodes most of the time.

The successfully trained DQN parameters is saved in the file `p1_dqn_agent.pth`. When the file was recorded the agent reached the goal at episode **330**.


## Learning Algorithm

The agent is trained using [Deep Q-Learning algorithm][dqn-paper] along with the following hyperparameters.

* Neural network for action-value estimation function

  - Network architecture
    - Input: 37 dimension vector
    - Hidden Layer 1
      - Linear with 256 nodes
      - Relu
    - Hidden Layer 2
      - Linear with 128 nodes
      - Relu
    - Output Layer
      - Linear with 4 nodes

  - Optimization algorithm
    - Adam
    
  - Learning rate
    - 5e-4

* Epsilon anealing for e-greedy policy

  - Start: 1.
  - End: 0.01
  - Decay: 0.995
  
* Discount factor, $\gamma$

  - 0.99
  
* Soft-update ratio, $\tau$

  - 0.001

* Network update interval

  - The local and target networks are updated every 4 time steps.
  
* Replay buffer size

  - 10^5
  
* Minibatch size

  - 64
  
With the above hyperparameters, the average score of the last 100 consecutive episodes reached 13.0 after 330 episodes.

[dqn-paper]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

## Ideas for Future WOrk

This project used the basic DQN algorithm. It can be improved by applying the following methods, which have been proved to overcome the weaknesses, especially the high bias of the basic DQN.

- double DQN
- dueling DQN
- prioritized experience replay
