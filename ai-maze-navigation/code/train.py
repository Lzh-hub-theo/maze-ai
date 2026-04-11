# train.py

from env import GridWorld
from agent import QLearningAgent
from config import *

def train():
    env = GridWorld()
    state_size = GRID_SIZE * GRID_SIZE
    action_size = 4
    rewards = []

    agent = QLearningAgent(state_size, action_size)

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward}")

        rewards.append(total_reward)

    return agent, rewards