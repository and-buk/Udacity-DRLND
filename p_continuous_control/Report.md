# Project Report

## Abstract
Using **actor-critic method** for reinforcement learning we trained a double-jointed arm to maintain its position at the target location for as many time steps as possible.

## Project Objective 

Implement an algorithm that allows to solve the environment task with an average score of +30 obtained from all agents (20) for 100 consecutive episodes.

## Introduction

Reinforcement Learning is about learning an optimal policy from interaction with the environment. With **value-based methods**, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function. The optimal policy is then obtained from the optimal action-value function estimate. **Policy-based methods** directly learn the optimal policy, without having to maintain a separate value function estimate. **Policy gradient methods** are a subclass of policy-based methods that estimate the weights of an optimal policy through gradient ascent.

**Strengths of policy-based methods**: 

 - **Simplicity**: The problem with value-based methods is that they can have a big oscillation while training. This is because the choice of action may change dramatically for an arbitrarily small change in the estimated action values. Policy-based methods directly get to the problem at hand (estimating the optimal policy). This tends to make them stable and reliable.

- **Stochastic policies**: Policy-based methods can learn true stochastic policies (a value-based methods tends to learn a deterministic or near deterministic policy).

- **Continuous action spaces**: Policy-based methods are well-suited for continuous and high-dimensional action spaces.

**Weakness of policy-based methods**:

- **Data-inefficient**: Policy-based methods are data-inefficient and rely on a large number of samples to learn a useful policy.

- **High variance**: A critical challenge of policy gradient methods is the high variance of the gradient estimator and therefore *slow convergence (slow learning)*. Policy gradient methods are able to compute an unbiased gradient, but suffer from high variance.

**Actor-critic methods** are able to trade-off between the strengths and weaknesses of value-based and policy-based approaches:
- They use value-based techniques to further reduce the variance of policy-based methods and increase the learning speed respectively.
- They have more consistent and smooth convergence than value-based methods, therefore more stable.
- Need fewer samples than policy-based methods.

## Algorithm Selection 

To solve version 2 of the environment with continuous action space we use **Deep Deterministic Policy Gradients (DDPG)** algorithm ([**more detail in paper**](https://arxiv.org/abs/1509.02971)) specifically adapted to work for multiple agents.

**DDPG** is a **model-free**, **off-policy**, **actor-critic algorithm**, which concurrently learns a deterministic policy and a Q-functon by using each to improve the other. It uses off-policy data and the Bellman equation to learn the Q-function (*policy evaluation - compute the value function for a policy*), and the Q-function to learn the policy (*policy improvement - use the value function to obtain a better policy*).

### Key components of **DDPG**
- **Actor-critic architecture** with two elements, actor and critic.
  - Actor: Parameterized function ***&mu;(s|&theta;<sup>&mu;</sup>)*** that specifies the current policy by deterministically mapping states to a specific action.
  - Critic: The critic ***Q(s, a|&theta;<sup>Q</sup>)*** is learned using the Bellman equation as in Q-learning.

The *deterministic* actor maximizes the output of the critic. Critic equates the value of the actor output (action) to returns observed in reality. 

- As in Deep Q-network to adapt the Q-learning algorithm in order to make effective use of large neural networks as function approximators and avoid instability of learning are used **two important techniques**:
  - **Experience Replay**: The actor and critic networks are trained off-policy with samples uniformly  from replay buffer to minimize correlations between samples.
  - Separate **target network**: The critic network is trained with both actor and critic target networks to give consistent target values during temporal difference backups.
- **Random noise process for action exploration**: Add noise sampled from *Ornstein-Uhlenbeck process* to actor policy.
- **“Soft” target updates**: The weights of targets networks are updated by having them slowly track the learned networks, so the target values are constrained to change slowly, greatly improving the stability of learning.
- Estimate the weights of an actor policy through gradient ascent.
- Estimate the weights of a critic network through gradient descent.
- **Batch normalization technique**: This technique normalizes each dimension across the samples in minibatch to have unit mean and variance. To minimize covariance shift during training, by ensuring that each network's layer receives whitened input.

### Amendments to **DDGP** algorithm (to make the code work with 20 agents)
- Each agent adds its experience to a replay buffer that is shared by all agents, and
- The (local) actor and critic networks are updated 10 times (`UPDATE_FREQ`) after every 20 timesteps (`UPDATE_EVERY`) in row (one for each agent), using different samples from the buffer.
- Using gradient clipping technique when training the critic network.

### Hyperparameters and model architecture description
For learning the neural network parameters we use `Adam algorithm` with a learning rate of 10<sup>-4</sup> and 10<sup>-3</sup> for the actor (`LR_ACTOR`) and critic (`LR_CRITIC`) networks respectively. For compute Q target we use a discount factor (`GAMMA`) of = 0.99. For the soft target updates we use &tau; (`TAU`) = 0.001. The neural networks use the rectified non-linearity for all hidden layers. Since every entry in the action must be a number between -1 and 1 we add a tanh activation function to the final output layer of the actor network. The actor network has 3 hidden layers with 89, 144 and 233 units respectively. The critic network has 2 hidden layers with 144 and 233 units respectively. We use *batch normalization* on all layers of both networks. The actor network to avoid overfitting has dropout layers with probability 0.1. 
Actions are not included until the 2nd hidden layer of critic network. The final layer weights and biases of both the actor and critic networks are initialized from a uniform distribution [-3 х 10<sup>-3</sup>, 3 х 10<sup>-3</sup>] to provide the initial outputs for the policy and value estimates are near zero. The other layers are initialized from uniform distributions [-1/<span class="radic"><sup><var></var></sup>√</span><span class="radicand"><var>f</var></span>, 1/<span class="radic"><sup><var></var></sup>√</span><span class="radicand"><var>f</var></span>] where *f* is the fan-in of the layer. We train with minibatch sizes (`BATCH_SIZE`) of 128 and use a replay buffer size (`BUFFER_SIZE`) of 1000000. For the exploration noise process we use an *Ornstein-Uhlenbeck process* with &theta; = 0.15 and &sigma; = 0.2. &sigma; evenly decreases from 0.2 to 0.008 in the range of received average reward from 30 to 40.

## Conclusion

<p align="left">
  <img src="https://github.com/and-buk/reinforcement-learning/blob/master/p_continuous_control/images/final_plot.png" width="550">
</p>
Environment solved in 100 episodes!	Average Score: 33.72

## Ideas for future

- Despite the fact that task is complete, the final plot doesn't look well. There is a high variance between average scores from episode to episode during training. The agent's behavior doesn't stable. To improve it, in my opinion, it is necessary to continue experimenting with architecture of both networks (reduce the number of hidden layers of actor network and/or hidden layers units, add dropout layers into critic network, etc.) and tuning hyperparameters.

- Implement more stable, without extreme brittleness and hyperparameter sensitivity off-policy **soft actor-critic (SAC)** algorithm that, in practice, exceeds both the efficiency and final performance of DDPG ([**more detail in paper**](https://arxiv.org/abs/1801.01290)).
