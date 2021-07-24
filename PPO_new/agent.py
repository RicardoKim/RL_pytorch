import torch
import random

import numpy as np
import pandas as pd
import os 
from collections import deque
from Model.MLPmodel import MLPAGENT
from Model.MLPmodel import MLPCRITIC
from Argument import argument
from torch.distributions import Normal
import torch.optim as optim
import torch.nn.functional as F

class agent(object):
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer = deque(maxlen = 2000)
        self.argument = argument()
        self.actor_network = MLPAGENT(env)
        self.critic_network  = MLPCRITIC(env)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr = self.argument.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr = self.argument.CRITIC_LEARNING_RATE)
        self.old_actor_network = MLPAGENT(env)
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        mu, std = self.actor_network(state)
        normal_distribution = Normal(mu, std)
        action = normal_distribution.sample()
        return action

    def unpack_batch(self):
        state= np.zeros((self.argument.BATCH_SIZE, self.actor_network.state_shape))
        next_state = np.zeros((self.argument.BATCH_SIZE, self.critic_network.state_shape))
        action = np.zeros((self.argument.BATCH_SIZE, self.actor_network.action_shape))
        reward = np.zeros((self.argument.BATCH_SIZE, 1))
        done = np.zeros((self.argument.BATCH_SIZE, 1))
        idx_batch = set(random.sample(range(len(self.buffer)), self.argument.BATCH_SIZE))
        for index, value in enumerate(idx_batch):
            state[index] = self.buffer[value][0]
            action[index] = self.buffer[value][2]
            next_state[index] = self.buffer[value][1]
            reward[index] = self.buffer[value][3]
            done[index] = self.buffer[value][4]
        return state, action, next_state, reward, done



    def train(self, render = False, env_name = "MountainCarContinuous-v0"):
        directory = "Log"
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_directory = "saved_model"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        index = 0
        log = pd.DataFrame(columns= ['episode_reward', 'actor_loss', 'critic_loss'], index = np.arange(0 , self.argument.TRAIN_EPOCH, 1))
        for episode in range(self.argument.TRAIN_EPOCH):
            episode_reward = 0
            state = self.env.reset()
            done = False    
            while done == False :
                # if render:
                #     self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # train_reward = (reward + 8.1) / 8.1
                self.buffer.append((state, next_state, action, reward, done))
                episode_reward += reward

                if(len(self.buffer) < self.argument.BATCH_SIZE):
                    continue
                else:
                    batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.unpack_batch()
                    batch_state = torch.FloatTensor(batch_state).to(self.device)
                    batch_action = torch.FloatTensor(batch_action).to(self.device)
                    batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
                    batch_reward = torch.FloatTensor(batch_reward).to(self.device)
                    batch_done = torch.FloatTensor(batch_done).to(self.device)
                    with torch.no_grad():
                        value_target = batch_reward + self.argument.GAMMA * (1 - batch_done) * self.critic_network(batch_next_state)
                        advantage = value_target - self.critic_network(batch_state)
                    mu, std = self.actor_network(batch_state)
                    normal_distribution = Normal(mu, std)
                    log_prob = normal_distribution.log_prob(batch_action)

                    old_mu, old_std = self.old_actor_network(batch_state)
                    old_normal_distribution = Normal(old_mu, old_std)
                    old_log_prob = old_normal_distribution.log_prob(batch_action)

                    ratios = torch.exp(log_prob - old_log_prob.detach())
                    
                    surr1 = ratios * advantage
                    surr2 = torch.clamp(ratios, 1-self.argument.EPS_CLIP, 1+self.argument.EPS_CLIP) * advantage
                    entropy_bonus = normal_distribution.entropy()

                    actor_loss = -torch.min(surr1, surr2) - 0.01*entropy_bonus
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    value_loss = F.mse_loss(value_target, self.critic_network(batch_state))
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()
                    self.buffer.clear()
                    self.old_actor_network.load_state_dict(self.actor_network.state_dict())

                state= next_state
            if episode % 10 == 0:
                print('Episode : ', episode, "Episode Reward", episode_reward)
            if episode % 1 == 0:
                log.iloc[index]['episode_reward'] = episode_reward
                log.iloc[index]['actor_loss'] = actor_loss.detach().numpy()
                log.iloc[index]['critic_loss'] = value_loss.detach().numpy()
                log.to_csv(directory + "/" + env_name + "_log.csv")
                torch.save(self.actor_network.state_dict(), model_directory + '/' + env_name+'_saved_model.para')
                index += 1 
            
