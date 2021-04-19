import gym
from agent import agent

env = gym.make("Pendulum-v0")
test_agent = agent(env)
test_agent.train()