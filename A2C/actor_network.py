import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F

# 현재 버전은 state가 1차원임을 가정하고 진행하였다.


class Actor_Network(nn.Module):
    def __init__(self, env):
        super(Actor_Network, self).__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.env = env
        self.state_shape = env.reset().shape[0]
        # Discrete action_space인지
        # Continuous action space인지 확인하기 위해
        # 다음과 같이 구성함.
        # 둘이 loss를 구성하는 방식이 다름

        if(env.action_space.__repr__()[0] == 'D'):
            self.Discrete = True
            self.Continuous = False
            #open AI gym에서 Discrete object가 shape 함수에 이슈가 있어서 어거지로 만들었는데 별로다
            self.action_shape = np.expand_dims(np.zeros_like(env.action_space.sample()), axis = -1).shape[0]
        else:
            self.Discrete = False
            self.Continuous = True
            self.action_shape = env.action_space.shape[0]

        # 들어오는 state의 형태가 2차원인지 1차원인지에 따라 
        # Network를 유연하게 구성하려고 다음과 같은 변수를 생성함
        if(env.reset().ndim == 1 ):
            self.MLP = True
            self.CNN = False
            self.MLP_Agent()
        else:
            self.MLP = False
            self.CNN = True
            #여기 구현은 나중에~~~
        


    def MLP_Agent(self):
        self.linear1 = nn.Linear(self.state_shape, 64).to(self.device, dtype = torch.float64)
        self.linear2 = nn.Linear(64, 64).to(self.device, dtype = torch.float64)
        self.mu_output = nn.Linear(64, self.action_shape).to(self.device, dtype = torch.float64)
        self.std_output = nn.Linear(64, self.action_shape).to(self.device, dtype = torch.float64)

    
    def forward(self, x):
        if(self.MLP == True):
            #continuous action에 대해서만 진행함
            mu, std = self.MLP_forward(x)
        else:
            pass
            #구현해야하는 곳
        return mu.detach(), std.detach()

    def MLP_forward(self, x):
        x = torch.from_numpy(x).detach().to(self.device, dtype = torch.float64)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.mu_output(x)
        std = torch.sigmoid(self.std_output(x))
        return mu, std
        
        