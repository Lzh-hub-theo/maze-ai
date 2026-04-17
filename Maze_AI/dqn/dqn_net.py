# dqn_net.py
# 增强版 DQNNet，三层，每层256单元，含BatchNorm和Dropout
import torch
import torch.nn as nn

class DQNNet(nn.Module):
    def __init__(self, input_dim, action_size, hidden_dim=256, dropout_p=0.2):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)
