import matplotlib.pyplot as plt
import pandas as pd
import os

def drawing_test_result():
    env_name = "CartPole-v1"

    directory = "PPO_train_result"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    run_num_pretrained = 1  
    csv_directory = "PPO_logs" + '/' + env_name + '/'
    checkpoint_path = csv_directory + "PPO_{}_log_{}.csv".format(env_name, run_num_pretrained)
    data = pd.read_csv(checkpoint_path)
    fig=plt.figure()
    plt.ion()
    plt.ioff()
    plt.plot(data['timestep'], data['reward'])
    plt.savefig(directory + "PPO_train_result_{}.png".format(run_num_pretrained))

if __name__ == "__main__":
    drawing_test_result()