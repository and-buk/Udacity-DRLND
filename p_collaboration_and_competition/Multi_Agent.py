import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 512         # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-2               # for soft update of target parameters
LR_ACTOR = 0.0001        # learning rate of the actor  
LR_CRITIC = 0.001        # learning rate of the critic
UPDATE_EVERY = 1         # how often to update the network
UPDATE_FREQ = 1          # how many times update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agents():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, seed, logger = None):
        """Initialize an Agents object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents to learn
            seed (int): seed
        """
        self.state_size = state_size
        self.action_size = action_size        
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * self.num_agents, action_size * self.num_agents, seed).to(device)
        self.critic_target = Critic(state_size * self.num_agents, action_size * self.num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
               
        self.agents = {}
        
        # Get local, target actor networks and optimizer for each agents 
        for num_agent in range(self.num_agents):
            self.agents[num_agent] = (self.actor_local, self.actor_target)
                        
        # Copy the weights from local to target networks
        for num_agent in range(self.num_agents):
            self.soft_update(self.agents[num_agent][0], self.agents[num_agent][1], 1)

        self.soft_update(self.critic_local, self.critic_target, 1)
              
        # Noise process for action exploration
        self.noise = OUNoise((num_agents, action_size), seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps if enough samples are available in memory 
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            # Get random subset and learn UPDATE_FREQ times
            for t in range(UPDATE_FREQ):
                experiences = self.memory.sample()                           
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True, noise=0.0):
        """Returns actions for given state as per current policy."""
        
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        
        for num_agent, state in enumerate(states):           
            self.agents[num_agent][0].eval()
            with torch.no_grad():
                action = self.agents[num_agent][0](state).cpu().data.numpy()
                actions[num_agent, :] = action                                      
            self.agents[num_agent][0].train()
            
        # Provide an exploration policy by adding noise sampled from a noise process
        if add_noise :
            actions += self.noise.sample() * noise
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (x, a, r, x', done) tuples 
            gamma (float): discount factor
        """
        for num_agent in range(self.num_agents):
            x, actions, rewards, next_x, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            # Get target network actions from all the agents
            
            # Splits 'x' into a 'num_agents' of states
            states = torch.chunk(x, self.num_agents, dim = 1)
            # Splits 'next_x' into a 'num_agents' of next states
            next_states = torch.chunk(next_x, self.num_agents, dim = 1)
            
            # Get reward for each agent
            rewards = rewards[:,num_agent].reshape(rewards.shape[0],1)
            dones = dones[:,num_agent].reshape(dones.shape[0],1)
          
            target_actions = []
            for num_agent in range(self.num_agents):
                next_action = self.agents[num_agent][1](next_states[num_agent])
                target_actions.append(next_action)
            target_actions = torch.cat(target_actions, dim=1)
            
            with torch.no_grad():
                Q_targets_next = self.critic_target(next_x, target_actions)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[num_agent] + (gamma * Q_targets_next * (1 - dones[num_agent]))
            # Compute critic loss
            Q_expected = self.critic_local(x, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss            
            actions_pred = []
            for num_agent in range(self.num_agents):
                action_pred = self.agents[num_agent][0](states[num_agent])
                actions_pred.append(action_pred)
            actions_pred = torch.cat(actions_pred, dim=1)
            actor_loss = -self.critic_local(x, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            # Perform gradient ascent (-actor_loss)
            actor_loss.backward()
            self.actor_optimizer.step()
                      
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
            
        for num_agent in range(self.num_agents):
            self.soft_update(self.agents[num_agent][0], self.agents[num_agent][1], TAU)
                   
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(shape)
        
        # How "strongly" the system reacts to perturbations (the "decay-rate" or "growth-rate")
        self.theta = theta
        
        # The variation or the size of the noise
        self.sigma = sigma 
        
        self.seed = np.random.seed(seed)
        self.reset()
        
    def set_sigma(self, value):
        self.sigma = value

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(x.shape) 
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["x", "action", "reward", "next_x", "done"])
    
    def add(self, x, action, reward, next_x, done):
        """Add a new experience to memory."""
        
        # Join a sequence of agents's states, next states and actions along columns
        x = np.concatenate(x, axis=0)
        next_x = np.concatenate(next_x, axis=0)
        action = np.concatenate(action, axis=0)
        
        e = self.experience(x, action, reward, next_x, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        x = torch.from_numpy(np.vstack([e.x for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_x = torch.from_numpy(np.vstack([e.next_x for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (x, actions, rewards, next_x, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)