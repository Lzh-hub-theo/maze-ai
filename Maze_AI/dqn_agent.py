# DQNAgent for Maze_AI
# Author: Copilot

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# 新增导入
from dqn.replay_buffer import ReplayBuffer
from dqn.dqn_net import DQNNet

class DQNAgent:
    def __init__(self, action_size, state_dim=2, device=None, learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.action_size = action_size
        self.memory = ReplayBuffer(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = 64
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.policy_net = DQNNet(state_dim, action_size, hidden_dim=256, dropout_p=0.2).to(self.device)
        self.target_net = DQNNet(state_dim, action_size, hidden_dim=256, dropout_p=0.2).to(self.device)
        self.update_target()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        self.target_update_freq = 500

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        import numpy as np
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.update_target()
        # epsilon 衰减移到 episode 结束后
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def get_best_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()
        return torch.argmax(q_values).item()

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target()
