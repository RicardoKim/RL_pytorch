import gym
import argparse
from PPO import agent

def main(args):
    env_name = args.env
    save_interval = int(args.interval)
    env = gym.make(env_name)
    ppo_agent = agent(env, save_interval)
    ppo_agent.train(env_name = env_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Result visualize.')
    parser.add_argument('--env', help="What do you want to visualize", default="Pendulum-v0")
    parser.add_argument('--interval', help="Interval that you want to save result", default= 10)
    args = parser.parse_args()
    main(args)
