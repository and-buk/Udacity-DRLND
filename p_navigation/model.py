import torch
import torch.nn as nn # neural networks
import torch.nn.functional as F # layers, activations and more

class Dueling_QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, 
                 state_size, 
                 action_size,  
                 fc1_units = 256, 
                 fc2_units = 128, 
                 fc3_units = 64
                 ):
        """Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
                state is 2-D tensor of shape (n, state_size)
            action_size (int): Dimension of each action
                action is 2-D tensor of shape (n, action_size)
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        # Calls the init function of nn.Module
        super(Dueling_QNetwork, self).__init__()
        self.common_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units), # fully connected layer
            nn.BatchNorm1d(fc1_units), # batch norm layer 
            nn.Dropout(p=0.5), # dropout layer for any dimensional input
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.BatchNorm1d(fc2_units),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(fc2_units, fc3_units),
            nn.BatchNorm1d(fc3_units), # batch norm layer 
            nn.Dropout(p=0.2),
            nn.ReLU()
            )
        # Get state-value
        self.value_func_layer = nn.Linear(fc3_units, 1)
        # Get advantages for each action
        self.adv_func_layer = nn.Linear(fc3_units, action_size)
              
    def forward(self, state):
        """Build a network that maps state -> action values.
        
        Params
        ======
            state (torch.Tensor): Dimension of each state
                2-D tensor of shape (n, state_size)
                
        Returns
        ====== 
            Q value (torch.Tensor): set of Q values, one for each action
                2-D tensor of shape (n, action_size)
        """
        x = self.common_layer(state)
        v_s = self.value_func_layer(x)
        a_sa = self.adv_func_layer(x)
        return v_s + (a_sa - a_sa.mean())