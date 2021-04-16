import gym
import torch
import numpy as np 
from critic_network import Critic_Network
from actor_network import Actor_Network

def main():
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


if __name__ == '__main__':
    main()
