import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt

def visualize(args):
    log_dir = os.getcwd() + "/Log"
    file_list = os.listdir(log_dir)
    result = pd.read_csv(log_dir + "/" + file_list[1])
    plt.plot(result[args.type])
    plt.xlabel("Episode")
    plt.ylabel(args.type)
    directory = "Result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/ " + args.type + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Result visualize.')
    parser.add_argument('--type', help="What do you want to visualize")

    args = parser.parse_args()
    visualize(args)
