import gym
import torch
import numpy as np 
from agent import agent
def main():
    env = gym.make("Pendulum-v0")
    a2c_agent = agent(env)
    a2c_agent.train()


if __name__ == '__main__':
    main()
