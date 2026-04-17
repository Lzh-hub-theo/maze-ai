#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 迷宫环境类 - 用于强化学习

import numpy as np
import mapp

class MazeEnv:
    # 奖励设置
    REWARD_GOAL = 200.0      # 到达终点奖励（提升）
    REWARD_STEP = -0.01      # 每步惩罚（减小）
    REWARD_WALL = -5.0       # 撞墙惩罚（加大）
    REWARD_REVISIT = -0.1    # 重复访问惩罚

    """基于mapp.py的大地图迷宫环境，状态为归一化(x, y)"""
    def __init__(self):
        self.map_list = [row[:] for row in mapp.map_list]
        self.maze_height = len(self.map_list)
        self.maze_width = len(self.map_list[0])
        # 统一主程序定义：9为起点，3为终点
        self.start_pos = None
        self.goal_pos = None
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                if self.map_list[y][x] == 9:
                    self.start_pos = (x, y)
                if self.map_list[y][x] == 3:
                    self.goal_pos = (x, y)
        if self.start_pos is None:
            self.start_pos = (self.maze_width-2, 1)  # 默认右上角附近
        if self.goal_pos is None:
            self.goal_pos = (1, self.maze_height-2)  # 默认左下角附近
        self.action_size = 4
        self.visited = set()
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        self.visited = set()
        self.visited.add(self.start_pos)   # 起点默认已访问
        return self._get_state()

    def _get_state(self):
        # 返回归一化坐标[x/W, y/H]
        x, y = self.agent_pos
        return np.array([x / (self.maze_width-1), y / (self.maze_height-1)], dtype=np.float32)

    def _is_walkable(self, x, y):
        if x < 0 or x >= self.maze_width or y < 0 or y >= self.maze_height:
            return False
        return self.map_list[y][x] != 1

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:
            new_x, new_y = x, y - 1
        elif action == 1:
            new_x, new_y = x, y + 1
        elif action == 2:
            new_x, new_y = x - 1, y
        elif action == 3:
            new_x, new_y = x + 1, y
        else:
            new_x, new_y = x, y
        if not self._is_walkable(new_x, new_y):
            # 撞墙
            return self._get_state(), self.REWARD_WALL, False
        self.agent_pos = (new_x, new_y)
        next_state = self._get_state()
        if self.agent_pos == self.goal_pos:
            # 到达终点
            return next_state, self.REWARD_GOAL, True
        # 普通步
         # 计算普通步奖励（含重复访问惩罚）
        reward = self.REWARD_STEP
        if self.agent_pos in self.visited:
            reward += self.REWARD_REVISIT
        else:
            self.visited.add(self.agent_pos)

        return next_state, reward, False

