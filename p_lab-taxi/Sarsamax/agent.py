import numpy as np
from collections import defaultdict

np.random.seed(7)

class Agent:

    def __init__(self, nA=6, alpha = 0.5, gamma = 0.85, start_epsilon = 1):
        """ Initialize agent.

        Params
        ======
        - nA (int): number of actions available to the agent
        - alpha (float): step-size parameter for the update step
        - gamma (float): discount rate (always between 0 and 1, inclusive)
        - epsilon (float): value of random action probability
        """
        self.nA = nA
        # initialize action-value function (empty dictionary of arrays)
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.start_epsilon = start_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_episode = 1

    def get_probs(self, state):
        """ Obtains the action probabilities corresponding to epsilon-greedy policy """
        # decrease epsilon with each episode
        epsilon = self.start_epsilon / self.num_episode
        # all possible action with equal probabilities
        policy_s = np.ones(self.nA) * epsilon / self.nA 
        policy_s[np.argmax(state)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s 
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = np.random.choice(np.arange(self.nA), p=self.get_probs(self.Q[state]))
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:          
            # estimate in Q-table for state, action pair
            old_esimate = self.Q[state][action]
            # alternative estimate for next state, action pair
            target = reward + self.gamma * np.max(self.Q[next_state])
            # error in estimate
            error = target - old_esimate
            # get updated Q(s,a) value
            self.Q[state][action] = old_esimate + self.alpha * error        
        else:
            self.num_episode +=1