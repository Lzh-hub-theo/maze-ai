# replay_buffer.py
# 经验回放池实现，maxlen=10000，支持多样性采样
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.memory = deque(maxlen=maxlen)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
