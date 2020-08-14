from collections import deque
import sys
import math
import numpy as np

def interact(env, agent, num_episodes=20000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        # agent selects an action A
        action = agent.select_action(state)
        while True:
            # agent performs the selected action A, observe R, S'
            next_state, reward, done, _ = env.step(action)
            # agent selects next action A'
            next_action = agent.select_action(next_state)
            # agent performs internal updates based on sampled experience (S-A-R-S'-A')
            agent.step(state, action, reward, done, next_state, next_action)
            # update the sampled reward
            samp_reward += reward
            # update the state (S <- S') to next time step
            state = next_state
            # update the action (A <- A') to next time step
            action = next_action
            if done:
                agent.step(state, action, reward, done)
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward