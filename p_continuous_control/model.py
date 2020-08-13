import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Calculate the range of values for uniform distributions
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, 
                 state_size, 
                 action_size, 
                 seed, 
                 fc1_units = 89, 
                 fc2_units = 144,
                 fc3_units = 233):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 =  nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units) # Batch Normalization 
        self.dp1 = nn.Dropout(p=0.1) # Dropout Layer
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units) # Batch Normalization 
        self.dp2 = nn.Dropout(p=0.1) # Dropout Layer
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units) # Batch Normalization 
        self.dp3 = nn.Dropout(p=0.1) # Dropout Layer
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.tanh = nn.Tanh()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Apply to layers the specified weight initialization
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.dp1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.dp3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        output = self.tanh(x)
        return output

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, 
                 state_size, 
                 action_size,
                 seed,
                 fc1_units=144,
                 fc2_units=233
                ): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.relu1 = nn.ReLU()
           
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(fc2_units, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Apply to layer the specified weight initialization
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)   
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.relu1(x)
        x = torch.cat((x, action), dim = 1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        output = self.fc3(x)
        return output