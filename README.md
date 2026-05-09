# Tank Battle - 深度强化学习路径规划项目

基于深度强化学习的智能路径规划系统，包含迷宫探索和坦克大战两个方向。

## 项目目录

| 目录 | 内容 | 状态 |
|------|------|------|
| `Maze_AI/` | PyTorch DQN迷宫训练系统（主要成功项目） | ✅ |
| `TEST/` | BFS与DQN对比演示系统 | ✅ |
| `ai-maze-navigation/` | Q-learning迷宫探索实现 | ✅ |
| `battle-city-cv/` | 通过OpenCV+Selenium坦克大战视觉识别网页端各个种类图像 | ✅ |
| `battle-city/` | DQN整合React坦克大战游戏让ai学会玩坦克大战游戏 | ❌ |

---

## 项目概述

本项目尝试用深度强化学习让AI学会两项任务：
1. **走迷宫**（成功）- AI通过DQN强化学习找到从起点到终点的最短路径
2. **打坦克大战**（失败）- 尝试让AI在坦克大战游戏中自动导航和战斗

---

## 成功项目：走迷宫

### 1. Maze_AI/run（主要项目）

基于 PyTorch + DQN 的强化学习迷宫探索系统。让ai学会走迷宫游戏。本项目的迷宫游戏参考了github开源项目[https://github.com/Wonz5130/Maze_AI.git](https://github.com/Wonz5130/Maze_AI.git)

项目收敛效果：
![alt text](<training-reward-curve.png>)

**核心代码**:
- `main_ai.py` - 主程序，包含训练和演示模式
- `maze_env.py` - 迷宫环境类，定义状态、动作、奖励
- `dqn_agent.py` - DQN智能体，包含经验回放、目标网络、epsilon-greedy
- `dqn/dqn_net.py` - 神经网络（3层全连接，无BatchNorm）
- `dqn/replay_buffer.py` - 经验回放缓冲区
- `score_map.py` - BFS计算的最短路径分数矩阵，用于reward shaping
- `mapp.py` - 迷宫地图定义

**技术细节**:
| 参数 | 值 |
|------|-----|
| 状态维度 | 2（归一化坐标x/y） |
| 动作空间 | 4（上、下、左、右） |
| 网络结构 | 2 → 256 → 256 → 256 → 4 |
| 学习率 | 1e-3 |
| gamma | 0.95 |
| epsilon衰减 | 0.99 |
| 目标网络更新频率 | 200步 |
| BatchSize | 64 |
| 经验回放池 | 10000 |

**奖励机制**:
- 到达终点: +100
- 撞墙: -2
- 普通步: 基于BFS分数矩阵的reward shaping

**解决的问题**:
- BatchNorm在RL中不稳定 → 移除BatchNorm/Dropout
- 梯度爆炸 → 使用Huber Loss
- 训练频率过高 → 每步学习改为稳定更新

### 2. TEST（辅助项目）

BFS与DQN对比演示系统，动画展示两种算法的搜索过程和效果对比。

**核心代码**:
- `main.py` - 四阶段演示（BFS找最优路径、DQN训练、DQN演示、对比）
- `env.py` - Gym风格的迷宫环境
- `dqn_agent.py` - DQN智能体实现
- `bfs.py` - BFS最短路径算法
- `visualizer.py` - Pygame可视化渲染
- `config.py` - 所有超参数配置

**特性**:
- 随机生成迷宫
- 实时动画展示搜索过程
- BFS vs DQN路径对比
- 支持视频导出

### 3. ai-maze-navigation（Q-learning项目）

基于 Q-learning 的迷宫探索实现。

**核心代码**:
- `code/main.py` - 主程序
- `code/train.py` - 训练逻辑
- `code/env.py` - 环境定义
- `code/visualize.py` - 结果可视化

---

## 失败项目：坦克大战

### battle-city

React实现的坦克大战游戏，基于shinima/battle-city复刻版。原项目github地址：[https://github.com/feichao93/battle-city.git](https://github.com/feichao93/battle-city.git)

**问题**: 只有游戏框架，没有实现AI控制。

### battle-city-cv

尝试通过OpenCV视觉识别 + Selenium控制来实现AI玩游戏。

**实现**: 实现了网页端坦克、砖墙等物体的识别

**问题**: 技术实现复杂，未能实现端到端的强化学习训练。

---

## 技术栈

- **Python**: PyTorch, Pygame, NumPy, OpenCV, Selenium
- **前端**: React, TypeScript, Redux, Redux-Saga