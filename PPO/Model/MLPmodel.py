from torch import nn
import torch.nn.functional as F


class MLPAGENT(nn.Module):
    def __init__(self, env):
        super(MLPAGENT, self).__init__()
        self.env = env
        self.state_shape = env.reset().shape[0]
        self.action_shape = env.action_space.shape[0]
        self.linear1 = nn.Linear(self.state_shape, 64)
        self.linear2 = nn.Linear(64, 256)
        self.mu_output = nn.Linear(256, self.action_shape)
        self.std_output = nn.Linear(256, self.action_shape)

    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = F.relu(self.mu_output(x))
        std = F.softplus(self.std_output(x)) + 1e-3
        return mu, std

class MLPCRITIC(nn.Module):
    def __init__(self, env):
        super(MLPCRITIC, self).__init__()
        self.env = env
        self.state_shape = env.reset().shape[0]
        self.action_shape = env.action_space.shape[0]
        self.linear1 = nn.Linear(self.state_shape, 64)
        self.linear2 = nn.Linear(64, 256)
        self.output = nn.Linear(256, 1)

    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        output = self.output(x)
        return output

        
        