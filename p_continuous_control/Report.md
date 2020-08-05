# Project Report

## Abstract
Using **actor-critic method** for reinforcement learning we trained a double-jointed arm to maintain its position at the target location for as many time steps as possible.

## Project Objective 

## Introduction

## Algorithm Selection 

To solve version 2 of the environment with continuous action space we use **Deep Deterministic Policy Gradients (DDPG)** algorithm ([**more detail in paper**](https://arxiv.org/abs/1509.02971)) specifically adapted to work for multiple agents.

**DDPG** is a model-free, off-policy actor-critic algorithm, which concurrently learns a deterministic policy and a Q-functon by using each to improve the other. It uses off-policy data and the Bellman equation to learn the Q-function, and the Q-function to learn the policy.

Key components of **DDPG**:
- **Actor-critic architecture** with two elements, actor and critic.
  - Actor: Parameterized function ***&mu;(s|&theta;<sup>&mu;</sup>)*** that specifies the current policy by deterministically mapping states to a specific action.
  - Critic: The critic ***Q(s, a|&theta;<sup>Q</sup>)*** is learned using the Bellman equation as in Q-learning.
- As in Deep Q-network to adapt the Q-learning algorithm in order to make effective use of large neural networks as function approximators and avoid instability of learning are used **two important techniques**:
  - **Experience Replay**: The actor and critic networks are trained off-policy with samples uniformly  from replay buffer to minimize correlations between samples.
  - Separate **target network**: The critic network is trained with both actor and critic target networks to give consistent target values during temporal difference backups.
- **Random noise process for action exploration**: Add noise sampled from *Ornstein-Uhlenbeck process* to actor policy.
- **“Soft” target updates**: The weights of targets networks are updated by having them slowly track the learned networks, so the target values are constrained to change slowly, greatly improving the stability of learning.
- Estimate the weights of an actor policy through gradient ascent.
- Estimate the weights of a critic network through gradient descent.
- **Batch normalization technique**: To minimize covariance shift during training, by ensuring that each network's layer receives whitened input.


