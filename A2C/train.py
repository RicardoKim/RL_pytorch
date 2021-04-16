import gym
import torch
import numpy as np 
import deque
from critic_network import Critic_Network
from actor_network import Actor_Network

class train(object):
    def __init__(self, env):
        self.env = env
        self.buffer = deque(maxlen = 2000)
def train(env):
    env = gym.make("Pendulum-v0")
    critic_network = Critic_Network(env)
    actor_network = Actor_Network(env)
    done = False
    state = env.reset()
    while(done == False):
        env.render()
        action = actor_network(state)
        value = critic_network(state)
        next_state, reward, done, info = env.step(action)
        state= next_state
