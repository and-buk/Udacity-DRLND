# Algorithm description

<p align="center">
  <img src="https://github.com/and-buk/reinforcement-learning/blob/master/p_lab-taxi/Sarsa/images/Sarsa_alg.png" width="750">
</p>

# Instructions

The repository contains three files:
- `agent.py`: Reinforcement learning agent here.
- `monitor.py`: The `interact` function tests how well agent learns from interaction with the environment.
- `main.py`: Run this file in the terminal to check the performance of agent.

Begin by running the following command in the terminal:
```
python main.py
```

When you run `main.py`, the agent that you specify in `agent.py` interacts with the environment for 20,000 episodes. The details of the interaction are specified in `monitor.py`, which returns two variables: `avg_rewards` and `best_avg_reward`.
- `avg_rewards` is a deque where `avg_rewards[i]` is the average (undiscounted) return collected by the agent from episodes `i+1` to episode `i+100`, inclusive.  So, for instance, `avg_rewards[0]` is the average return collected by the agent over the first 100 episodes.
- `best_avg_reward` is the largest entry in `avg_rewards`. This is the final score that you should use when determining how well agent performed in the task.
