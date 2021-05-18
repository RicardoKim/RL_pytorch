import os
import glob
import time
from datetime import datetime
import pandas as pd
import torch
import numpy as np

import gym

# import pybullet_envs

from PPO import PPO



################################### Training ###################################

def train():

    print("============================================================================================")


    ####### initialize environment hyperparameters ######

    env_name = "CartPole-v1"


    update_timestep = 200      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)

    #####################################################



    print("training environment name : " + env_name)

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.n

    print("State dimension : " + str(state_dim))
    print("Action dimension : " + str(action_dim))

    directory = "PPO_save"
    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}.pth".format(env_name)
    print("save checkpoint path : " + checkpoint_path)


    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)


    print_running_reward = 0
    print_running_episodes = 0


    time_step = 0
    i_episode = 0

    succeed = False

    log = pd.DataFrame(columns = ['Epi', 'Reward'])

    epi = 0
    time_step = 0
    pass_time = 0
    while((epi < 1000) & (not succeed)):
        state = env.reset()
        epi_reward = 0
        done = False
        while not done:
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step +=1
            epi_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update()
        print("Episode : {} \t\t Reward : {}".format(epi, epi_reward))
        log.loc[epi] = [epi, epi_reward]
        log.to_csv(directory + "log.csv")
        epi += 1

        if(epi_reward == 500):
            pass_time += 1
        else:
            pass_time = 0
        
        if(pass_time == 10):
            succeed = True
        ppo_agent.save(checkpoint_path)
    log.to_csv(directory + "log.csv")
    env.close()

    return


if __name__ == '__main__':

    train()
    
    
    
    
    
    