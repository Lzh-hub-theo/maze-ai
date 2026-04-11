# main.py

from train import train
from env import GridWorld
from visualize import plot_path
import matplotlib.pyplot as plt

def main():
    agent, rewards = train()

    # 📈 画训练曲线
    plt.plot(rewards)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    env = GridWorld()
    plot_path(env, agent)

if __name__ == "__main__":
    main()