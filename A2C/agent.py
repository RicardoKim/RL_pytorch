import gym
import torch
import numpy as np 
from collections import deque
from critic_network import Critic_Network
from actor_network import Actor_Network
from Argument import argument
import torch.optim as optim

class agent(object):
    def __init__(self, env):
        self.env = env
        self.buffer = deque(maxlen = 2000)
        self.argument = argument()
        self.critic_network = Critic_Network(env)
        self.actor_network = Actor_Network(env)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr = self.argument.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr = self.argument.CRITIC_LEARNING_RATE)

    def compute_advantage_td_target(self, reward, state_value, next_state_value, done):
        if done:
            td_target = reward
            advantage = td_target - state_value
            advantage = advantage.detach().numpy()
            td_target = td_target.detach().numpy().reshape(-1, )
        else:
            td_target = reward + self.argument.GAMMA * next_state_value
            advantage = td_target - state_value
            advantage = advantage.detach()
            td_target = td_target.detach()

        return  advantage, td_target
    
    def get_action(self, state):
        mu, std = self.actor_network(state)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def unpack_batch(self):
        state= np.zeros((self.argument.BATCH_SIZE, self.actor_network.state_shape))
        action = np.zeros((self.argument.BATCH_SIZE, self.actor_network.action_shape))

        td_targets = np.zeros((self.argument.BATCH_SIZE, self.critic_network.action_shape))
        advantages = np.zeros((self.argument.BATCH_SIZE, self.critic_network.action_shape))

        for i in range(len(self.buffer)):
            state[i] = self.buffer[i][0]
            action[i] = self.buffer[i][1]
            td_targets[i] = self.buffer[i][2]
            advantages[i] = self.buffer[i][3]
        return state, action, td_targets, advantages

    def log_pdf(self, mu, std, action):
        
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * np.log(var * 2 * np.pi)
        log_policy_pdf = torch.tensor(log_policy_pdf, dtype=torch.float64 , device = self.actor_network.device)
        return torch.sum(log_policy_pdf, axis = 1).reshape(-1, 1)


    def train(self, render = False):
        done = False
        state = self.env.reset()

        for episode in range(self.argument.TRAIN_EPOCH):
            episode_reward = 0    
            while(done == False):
                if render:
                    self.env.render()
                action = self.get_action(state).detach().numpy()
                next_state, reward, done, info = self.env.step(action)

                state_value = self.critic_network(state)
                next_state_value = self.critic_network(next_state)
                advantage, td_target = self.compute_advantage_td_target(reward, state_value, next_state_value, done)
                
                self.buffer.append((state, action, td_target, advantage))
                episode_reward += reward

                if(len(self.buffer) < self.argument.BATCH_SIZE):
                    continue
                else:
                    self.actor_train()
                    self.critic_train()
                    self.buffer.clear()
                state= next_state
            print('Episode : ', episode, "Episode Reward", episode_reward.numpy)
            
    

    
    def actor_train(self):
        states, actions, _, advantages = self.unpack_batch()
        advantages = torch.tensor(advantages, dtype=torch.float64 , device = self.actor_network.device, requires_grad = True)
        self.actor_network.train()
        mu_s, std_s = self.actor_network(states)
        log_policy_pdf = self.log_pdf(mu_s.detach().numpy(), std_s.detach().numpy(), actions)
        loss_policy = torch.multiply(advantages, log_policy_pdf)
        loss = - torch.sum(loss_policy)
        loss.backward()
        self.actor_optimizer.step()

    def critic_train(self):
        states, _, td_targets, _ = self.unpack_batch()
        state_values = self.critic_network(states)
        td_targets = torch.tensor(td_targets)
        diff = (td_targets - state_values)
        loss = torch.mean(0.5 * torch.square(diff))
        loss.backward()
        self.critic_optimizer.step()
