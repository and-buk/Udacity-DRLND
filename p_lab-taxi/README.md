# Mini project: Taxi problem
([**Deep Reinforcement Learning Nanodegree**](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) coding exercise)

## Introduction

We use OpenAI Gym's **Taxi-v3 environment** ([**the code here**](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py)) and Temporal-Difference (TD) control methods to teach a taxi agent to navigate a small gridworld.

The Reinforcement Learning task (Taxi problem) based on [**this paper**](https://arxiv.org/pdf/cs/9905014.pdf).

### The environment description

There are 4 designated locations in the grid world indicated by `R(ed)`, `G(reen)`, `Y(ellow)`, and `B(lue)`. When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.

There are 6 possible actions, corresponding to moving `North`, `East`, `South`, or `West`, `picking up` the passenger, and `dropping off` the passenger.

As a taxi driver, you need to pick up and drop off passengers as fast as possible. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.

### Background

As we know to define a reinforcement learning task, we generally use Markov Decision Process (MDP) to model the environment. The MDP specifies the rules that the environment uses to respond to the agent's actions, including how much reward to give to the agent in response to its behavior. The agent's goal is to learn how to play by the rules of the environment, in order to maximize reward. In other words agent needs to find the optimal policy.

Temporal-Difference (TD) control methods: `Sarsa`, `Sarsamax (Q-learning)`, `Expected Sarsa` or value iteration algorithms estimate the action-value function by using the Bellman equation as an iterative update. All of them converge to the optimal action-value function (and so yield the optimal policy)

As the environment has small state spaces we represent the optimal action-value function in table with one row for each state and one column for each action (Q-table).

Using information about agent-environment interaction (states, actions and rewards) and presented below update rules we change Q-table values. 

<p align="center">
  <img src="https://github.com/and-buk/reinforcement-learning/blob/master/p_lab-taxi/images/Sarsa.png" width="650">
  <em> Sarsa </em>
</p>

<p align="center">
  <img src="https://github.com/and-buk/reinforcement-learning/blob/master/p_lab-taxi/images/Sarsamax.png" width="650">
  <em> Sarsamax (Q-learning) </em>
</p>

<p align="center">
  <img src="https://github.com/and-buk/reinforcement-learning/blob/master/p_lab-taxi/images/ExpSarsa.png" width="730">
  <em> Expected Sarsa </em>
</p>

To construct the optimal policy, we select the entries that maximize the action-value function (Q-table values) for each row (or state).

### Differences

- `Sarsa` and `Expected Sarsa` are both **on-policy** TD control algorithms. In this case, the same (*&straightepsilon;*-greedy) policy that is evaluated and improved also used to select actions.
- `Sarsamax` is **off-policy** method, where the (greedy) policy that is evaluated and improved is different from the (*&straightepsilon;*-greedy) policy that is used to select actions.  

## Getting started

The repository contains three sub-repositories with algorithms implementation, one for each TD control methods.


Install packages required to working the code in each sub-repository:
- Python 3.6 and higher
- `gym` (toolkit for developing and comparing reinforcement learning algorithms)
- NumPy, Matplotlib

Follow the instructions inside each repository.

## Analyzing perfomance

| TD control method    | Mean |  SD | Min.| Max.|
| :---:                |:---: |:---:|:---:|:---:|
| Sarsa                | 8.83 | 0.08|8.64 |9.05 |
| Sarsamax (Q-learning)| 8.84 | 0.1 |8.63 |9.05 |
| Expected Sarsa       | 8.75 | 0.13|8.44 |9.17 |

<p align="left">
  <img src="https://github.com/and-buk/reinforcement-learning/blob/master/p_lab-taxi/images/Histogram.png" width="550">
<p align="left">  
  <em> The frequency distribution histogram of average rewards for 100 trials (each trial = 20000 episodes) </em>
</p>
