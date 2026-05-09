# dqn_net.py
# 简化版 DQNNet，无 BatchNorm（RL 中 BatchNorm 不稳定）
import torch
import torch.nn as nn

class DQNNet(nn.Module):
    def __init__(self, input_dim, action_size, hidden_dim=256):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_size)
        # 使用 ReLU inplace 节省内存
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
