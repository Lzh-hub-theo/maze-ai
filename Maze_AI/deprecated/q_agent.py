#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Q-Learning Agent

import numpy as np

class QLearningAgent:
    """Q-learning智能体"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q表：存储每个状态-动作对的Q值
        self.q_table = np.zeros((state_size, action_size))
        
        # 超参数
        self.alpha = 0.1          # 学习率
        self.gamma = 0.9          # 折扣因子
        self.epsilon = 1.0        # 初始探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.1    # 最小探索率

    def save(self, filepath):
        """保存Q表到本地文件（npy格式）"""
        import numpy as np
        np.save(filepath, self.q_table)

    def load(self, filepath):
        """从本地文件加载Q表"""
        import numpy as np
        self.q_table = np.load(filepath)
    
    def choose_action(self, state):
        """
        选择动作（ε-greedy策略）
        state: 当前状态编号
        """
        if np.random.rand() < self.epsilon:
            # 探索：随机动作
            return np.random.randint(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        """
        Q-learning更新
        state: 当前状态
        action: 执行的动作
        reward: 获得的奖励
        next_state: 下一个状态
        """
        best_next_action_value = np.max(self.q_table[next_state])
        
        # Q-learning更新公式
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next_action_value - self.q_table[state][action]
        )
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_best_action(self, state):
        """获取当前状态下Q值最大的动作（纯利用，不探索）"""
        return np.argmax(self.q_table[state])
