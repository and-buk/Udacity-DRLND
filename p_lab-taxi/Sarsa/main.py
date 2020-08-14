from agent import Agent
from monitor import interact
import gym
import numpy as np

np.random.seed(7)

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)