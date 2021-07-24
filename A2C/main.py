import gym
import argparse
from agent import agent

def main(args):
    env_name = args.env
    env = gym.make(env_name)
    a2c_agent = agent(env)
    a2c_agent.train(env_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Result visualize.')
    parser.add_argument('--env', help="What do you want to visualize", default="Pendulum-v0")

    args = parser.parse_args()
    main(args)
