import torch
import torch.nn as nn
import torch.nn.functional as F

class NN3_prior(nn.Module):
    def __init__(self):
        super(NN3_prior, self).__init__()
        
        hidden_size = 20
        self.network = nn.Sequential(nn.Linear(9, hidden_size), 
                                     nn.Tanh(), 
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, 1))
        
    def forward(self, x):
        x = self.network(x)
        return x 
    
class NN4_prior(nn.Module):
    def __init__(self):
        super(NN4_prior, self).__init__()
        
        hidden_size = 30
        self.network = nn.Sequential(nn.Linear(9, hidden_size), 
                                     nn.Tanh(), 
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, 1))
        
    def forward(self, x):
        x = self.network(x)
        return x 
    
class NN5_prior(nn.Module):
    def __init__(self):
        super(NN5_prior, self).__init__()
        
        hidden_size = 40
        self.network = nn.Sequential(nn.Linear(9, hidden_size), 
                                     nn.Tanh(), 
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, hidden_size), 
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, 1))
        
    def forward(self, x):
        x = self.network(x)
        return x 