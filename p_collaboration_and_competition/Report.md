# Project Report

## Algorithm Selection 

To solve the environment task which require multiple agents to work together we use **multi-agent reinforcement learning (MARL) method** such as **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm ([**more detail in paper**](https://arxiv.org/abs/1706.02275)). 
**MADDPG** is multi-agent actor-critic algorithm for mixed cooperative-competetive environments. **MADDPG** is an extension of **DDPG** ([**more detail in paper**](https://arxiv.org/abs/1509.02971)) to the multi-agent setting.

Learning algorithm is presented below:

<p align="left">
  <img src="https://github.com/and-buk/Udacity-DRLND/blob/master/p_collaboration_and_competition/images/MADDPG_alg.png" width="700">
</p>

### Key components of **MADDPG for N agents**
- **Actor-critic architecture** with two elements, actor and critic.
  - Deterministic actor (policy) **for each agent**, where actor function only has access to local information (i.e. agent's own observations).
  - Centralized critic (action-value) function that explicitly uses the dicision-making policies (a<sub>1</sub>,..., a<sub>*N*;</sub>) of each agent in addition to the all their
  observation (x, x').
- As in Deep Q-network to adapt the Q-learning algorithm in order to make effective use of large neural networks as function approximators and avoid instability of learning are used **two important techniques**:
  - ***Shared* Experience Replay**: The actor and critic networks are trained off-policy with samples uniformly  from replay buffer to minimize correlations between samples.
  - Separate **target network**: The critic network is trained with both actor and critic target networks to give consistent target values during temporal difference backups.
- **Random noise process for action exploration**: Add noise sampled from *Ornstein-Uhlenbeck process* to actor policy.
- **“Soft” target updates**: The weights of targets networks are updated by having them slowly track the learned networks, so the target values are constrained to change slowly, greatly improving the stability of learning.
- Estimate the weights of an actor policy through gradient ascent.
- Estimate the weights of a critic network through gradient descent.

### Hyperparameters and model architecture description
For learning the neural network parameters we use `Adam algorithm` with a learning rate of 10<sup>-4</sup> and 10<sup>-3</sup> for the actor (`LR_ACTOR`) and critic (`LR_CRITIC`) networks respectively. 
For compute Q target we use a discount factor (`GAMMA`) of = 0.99. For the soft target updates we use &tau; (`TAU`) = 0.001. The neural networks use the rectified non-linearity for all hidden layers. 
Since every entry in the action must be a number between -1 and 1 we add a tanh activation function to the final output layer of the actor network. 
The both actor and critic networks have 2 hidden layers with 200 and 150 units . 
Actions are included before the 1nd hidden layer of critic network. The final layer weights and biases of both the actor and critic networks are initialized from a uniform distribution [-3 х 10<sup>-3</sup>, 3 х 10<sup>-3</sup>] to provide the initial outputs for the policy and value estimates are near zero. 
The other layers are initialized from uniform distributions [-1/<span class="radic"><sup><var></var></sup>√</span><span class="radicand"><var>f</var></span>, 1/<span class="radic"><sup><var></var></sup>√</span><span class="radicand"><var>f</var></span>] where *f* is the fan-in of the layer. 
We train during 5000 episodes with minibatch sizes (`BATCH_SIZE`) of 250 and use a replay buffer size (`BUFFER_SIZE`) of 100000.
For the exploration noise process we use an *Ornstein-Uhlenbeck process* with &theta; = 0.15 and &sigma; = 0.2.

## Conclusion

<p align="left">
  <img src="https://github.com/and-buk/Udacity-DRLND/blob/master/p_collaboration_and_competition/images/final_plot.png" width="550">
</p>
Environment solved in 1378 episodes!	Average Score: 0.50

## Ideas for future

- Implement **multi-agent TD3 (MATD3)** algorithm ([**more detail in paper**](https://arxiv.org/abs/1910.01465)).
- Experiment with **Parameter Space Noise for Exploration** ([**more detail in paper**](https://arxiv.org/abs/1706.01905))
- Use **Prioritized Experience Replay** ([**more detail in paper**](https://arxiv.org/abs/1511.05952))
