# Maze_AI — Code Wiki

## 一、项目概述

**Maze_AI** 是一个基于 **PyTorch + DQN（Deep Q-Network）** 强化学习算法的迷宫游戏项目。AI 智能体通过训练学习如何在迷宫中从起点走到终点，同时提供手动操作模式供玩家体验。

- **原始项目**：基于 [Wonz5130/Maze_AI](https://github.com/Wonz5130/Maze_AI.git) 的迷宫游戏进行二次开发
- **核心目标**：让 AI 通过强化学习自动学会走迷宫
- **许可证**：MIT License

---

## 二、项目架构总览

```
Maze_AI/
├── run/                        # 主运行目录（当前活跃代码）
│   ├── main_ai.py              # 主入口：训练 + AI演示 + 手动模式
│   ├── maze_env.py             # 迷宫强化学习环境（Gym风格接口）
│   ├── dqn_agent.py            # DQN智能体实现
│   ├── dqn/                    # DQN子模块
│   │   ├── dqn_net.py          # DQN神经网络结构
│   │   └── replay_buffer.py    # 经验回放池
│   ├── mapp.py                 # 大地图迷宫数据（43×42）
│   ├── score_map.py            # BFS最短路径分数矩阵（奖励塑形）
│   ├── manhattan_distance.py   # 曼哈顿距离计算（辅助工具）
│   ├── color.py                # 颜色常量定义
│   ├── test_movement.py        # 移动规则测试（命令行版）
│   ├── test_movement_gui.py    # 移动规则测试（GUI版）
│   ├── simhei.ttf              # 中文字体文件
│   ├── user.png                # 玩家角色图片
│   ├── statistic/              # 训练统计输出
│   │   ├── dqn_model.pth       # 已训练的DQN模型权重
│   │   └── reward_curve.png    # 训练奖励曲线图
│   └── trouble/                # 开发问题记录
├── deprecated/                 # 废弃的旧版代码
│   ├── main.py                 # 旧版主程序（随机迷宫 + 键盘/鼠标AI）
│   ├── main_new.py             # 旧版主程序（自定义迷宫）
│   ├── main_new1.py            # 旧版主程序（小地图迷宫）
│   ├── maze.py                 # 旧版迷宫生成器（DFS随机生成）
│   ├── mapp1.py                # 旧版小地图数据（6×6）
│   └── q_agent.py              # 旧版Q-Learning智能体
├── img/                        # 项目截图
├── bad_smell/                  # 代码坏味道记录
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
└── LICENSE                     # MIT许可证
```

---

## 三、核心模块详解

### 3.1 主入口 — `run/main_ai.py`

项目的主运行文件，负责初始化 Pygame 窗口、创建环境和智能体、训练/加载模型、以及游戏主循环。

#### 常量定义

| 常量 | 值 | 说明 |
|------|------|------|
| `SCREEN_WIDTH` | 800 | 窗口宽度 |
| `SCREEN_HEIGHT` | 800 | 窗口高度 |
| `ROOM_SIZE` | 15 | 每个迷宫格子的像素大小 |
| `TRAINING_EPISODES` | 500 | DQN训练轮数 |
| `FONT_PATH` | 绝对路径 | 中文字体文件路径 |
| `USER_PATH` | 绝对路径 | 玩家角色图片路径 |

#### 动作映射

```python
ACTION_TO_DIRECTION = {
    0: (0, -1),  # 上
    1: (0, 1),   # 下
    2: (-1, 0),  # 左
    3: (1, 0)    # 右
}
```

#### 关键函数

| 函数 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `print_text(font, x, y, text, color_val, shadow)` | 字体对象、坐标、文本、颜色、是否阴影 | 无 | 在屏幕上渲染文本，支持阴影效果 |
| `train_agent(env, agent, episodes)` | 环境、智能体、训练轮数 | 无 | 执行DQN训练循环，每轮结束后衰减epsilon，训练完成后用matplotlib绘制reward曲线并保存 |
| `action_to_room_change(roomx, roomy, action)` | 当前坐标、动作 | `(new_x, new_y)` | 根据动作计算新坐标 |
| `is_valid_move(roomx, roomy, action, r_list)` | 当前坐标、动作、地图 | `bool` | 检查移动是否合法（不越界、不撞墙） |
| `play_ai_mode(env, agent, r_list)` | 环境、智能体、地图 | `bool` | AI自动演示模式，智能体根据训练好的策略走迷宫 |
| `play_manual_mode(r_list)` | 地图 | `bool` | 手动操作模式，玩家用方向键走迷宫 |
| `show_menu()` | 无 | `'ai'` / `'manual'` / `None` | 显示主菜单，等待用户选择 |

#### 主流程

```
启动 → 初始化Pygame → 创建MazeEnv和DQNAgent
    → 检查已有模型 → 有则加载，无则训练并保存
    → 加载地图数据 → 解析起点(9)和终点(3)
    → 菜单循环 → AI演示 / 手动操作 / 退出
```

---

### 3.2 迷宫环境 — `run/maze_env.py`

实现了 Gym 风格的迷宫强化学习环境，是 AI 与迷宫交互的核心接口。

#### 类：`MazeEnv`

**初始化参数**：无（从 `mapp.py` 读取地图）

**核心属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `map_list` | `list[list[int]]` | 迷宫矩阵副本 |
| `maze_height` | `int` | 迷宫行数（42） |
| `maze_width` | `int` | 迷宫列数（43） |
| `start_pos` | `tuple(int, int)` | 起点坐标（值为9的格子） |
| `goal_pos` | `tuple(int, int)` | 终点坐标（值为3的格子） |
| `action_size` | `int` | 动作空间大小（4：上下左右） |
| `visited` | `set` | 已访问位置集合 |
| `agent_pos` | `tuple(int, int)` | 智能体当前位置 |

**奖励常量**：

| 常量 | 值 | 说明 |
|------|------|------|
| `REWARD_GOAL` | 100.0 | 到达终点奖励 |
| `REWARD_STEP` | 0 | 每步基础惩罚 |
| `REWARD_WALL` | -2 | 撞墙惩罚 |
| `REWARD_REVISIT` | 0 | 重复访问惩罚 |

**关键方法**：

| 方法 | 说明 |
|------|------|
| `reset()` | 重置环境到起点，清空已访问集合，返回初始状态 |
| `_get_state()` | 返回归一化坐标 `[x/W, y/H]`，作为神经网络输入 |
| `_is_walkable(x, y)` | 判断坐标是否可通行（不越界且非墙） |
| `step(action)` | 执行动作，返回 `(next_state, reward, done)` |

**奖励塑形机制（Reward Shaping）**：

`step()` 方法中使用了基于 BFS 最短路径分数矩阵的潜在奖励塑形：

```python
potential_old = score_matrix[old_y][old_x]
potential_new = score_matrix[new_y][new_x]
temp = potential_old - potential_new
if temp < 0:
    shaping = temp * 2    # 远离终点时加倍惩罚
else:
    shaping = temp        # 靠近终点时正常奖励
reward = REWARD_STEP + shaping
```

- 当 `potential_old > potential_new`（靠近终点）：给予正的塑形奖励
- 当 `potential_old < potential_new`（远离终点）：给予加倍的负塑形惩罚
- 这引导智能体更快地学习到正确的方向

---

### 3.3 DQN智能体 — `run/dqn_agent.py`

实现了完整的 DQN（Deep Q-Network）算法，包含双网络结构和经验回放。

#### 类：`DQNAgent`

**初始化参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_size` | 必填 | 动作空间大小 |
| `state_dim` | 2 | 状态维度（归一化x, y坐标） |
| `device` | 自动选择 | 计算设备（cuda/cpu） |
| `learning_rate` | 1e-3 | 学习率 |
| `gamma` | 0.95 | 折扣因子 |
| `epsilon` | 1.0 | 初始探索率 |
| `epsilon_decay` | 0.99 | 探索率衰减系数 |
| `epsilon_min` | 0.01 | 最小探索率 |

**核心属性**：

| 属性 | 说明 |
|------|------|
| `policy_net` | 策略网络（DQNNet），用于选择动作和训练 |
| `target_net` | 目标网络（DQNNet），用于计算目标Q值，定期从策略网络同步 |
| `memory` | 经验回放池（ReplayBuffer，容量10000） |
| `optimizer` | Adam优化器 |
| `loss_fn` | SmoothL1Loss（Huber损失），比MSE更稳定 |
| `batch_size` | 64 |
| `target_update_freq` | 200（每200步同步目标网络） |

**关键方法**：

| 方法 | 说明 |
|------|------|
| `update_target()` | 将策略网络权重复制到目标网络 |
| `remember(state, action, reward, next_state, done)` | 将经验存入回放池 |
| `choose_action(state)` | ε-greedy策略选择动作：以ε概率随机探索，否则选Q值最大的动作 |
| `learn()` | 从回放池采样batch_size个经验，计算DQN损失并反向传播更新策略网络 |
| `decay_epsilon()` | 每个episode结束后衰减探索率：`epsilon *= epsilon_decay` |
| `get_best_action(state)` | 纯利用模式，选择Q值最大的动作（无探索） |
| `save(filepath)` | 保存策略网络权重到.pth文件 |
| `load(filepath)` | 从.pth文件加载策略网络权重，并同步到目标网络 |

**DQN学习流程**：

```
1. 从回放池采样 batch_size 个 (s, a, r, s', done)
2. 计算当前Q值: Q(s, a) = policy_net(s)[a]
3. 计算目标Q值: target = r + γ * max(target_net(s')) * (1 - done)
4. 计算损失: SmoothL1Loss(Q(s,a), target)
5. 反向传播更新策略网络
6. 每200步同步目标网络
```

---

### 3.4 DQN神经网络 — `run/dqn/dqn_net.py`

#### 类：`DQNNet(nn.Module)`

4层全连接神经网络，无 BatchNorm（强化学习中 BatchNorm 不稳定）。

**网络结构**：

```
输入 (state_dim=2)
    → Linear(2, 256) → ReLU
    → Linear(256, 256) → ReLU
    → Linear(256, 256) → ReLU
    → Linear(256, action_size=4)
输出 (Q值: 4个动作的Q值估计)
```

| 参数 | 说明 |
|------|------|
| `input_dim` | 输入维度（2：归一化x, y坐标） |
| `action_size` | 输出维度（4：上下左右） |
| `hidden_dim` | 隐藏层维度（256） |

---

### 3.5 经验回放池 — `run/dqn/replay_buffer.py`

#### 类：`ReplayBuffer`

基于 `collections.deque` 实现的固定容量经验回放池。

| 方法 | 说明 |
|------|------|
| `add(state, action, reward, next_state, done)` | 添加一条经验 |
| `sample(batch_size)` | 随机采样指定数量的经验 |
| `__len__()` | 返回当前池中经验数量 |

- **最大容量**：10000（超出后自动丢弃最早的经验）
- **采样方式**：均匀随机采样

---

### 3.6 迷宫地图数据 — `run/mapp.py`

定义了 43 列 × 42 行的大迷宫矩阵 `map_list`。

**地图编码**：

| 值 | 含义 |
|------|------|
| `0` | 通路 |
| `1` | 墙壁 |
| `3` | 终点 |
| `9` | 起点 |

- 起点 `(9)` 位于第9行第42列（右下角区域）
- 终点 `(3)` 位于第9行第0列（左下角区域）

---

### 3.7 BFS分数矩阵 — `run/score_map.py`

预计算的 `score_matrix`，每个可通行格子的值表示从该位置到终点的最短步数。墙壁位置的值为 444（比最远可达距离大100+，确保不可达区域产生负奖励）。

**生成方式**：通过 `manhattan_distance.py` 中的 `compute_distance_score()` 函数，使用 BFS 从终点出发计算。

**用途**：在 `maze_env.py` 的 `step()` 方法中用于奖励塑形（Reward Shaping），引导智能体朝终点方向移动。

---

### 3.8 距离计算工具 — `run/manhattan_distance.py`

#### 函数：`compute_distance_score(maze)`

使用 BFS（广度优先搜索）从终点出发，计算迷宫中每个可通行位置到终点的最短步数。

**算法流程**：
1. 在迷宫中找到终点坐标（值为3的格子）
2. 从终点开始 BFS，逐层扩展
3. 墙壁(1)不可通过，其他(0, 3, 9)均视为可通行
4. 不可达位置设为 `INFINITE_DIST`（最远可达距离 + 100）
5. 返回二维分数矩阵

**独立运行**：可直接执行 `python manhattan_distance.py` 打印格式化的 `score_matrix`。

---

### 3.9 颜色常量 — `run/color.py`

定义了 15 种常用 RGB 颜色常量，供 Pygame 渲染使用：

`Aqua`, `Black`, `Blue`, `Fuchsia`, `Gray`, `Green`, `Lime`, `Maroon`, `NavyBlue`, `Olive`, `Purple`, `Red`, `Silver`, `Teal`, `White`, `Yellow`

---

### 3.10 测试文件

#### `run/test_movement.py` — 命令行版移动测试

测试迷宫环境的移动规则，执行固定动作序列（左1步→上2步→下20步→左2步→右2步），在命令行输出每步的位置变化和奖励。

#### `run/test_movement_gui.py` — GUI版移动测试

与命令行版功能相同，但使用 Pygame 窗口可视化展示移动过程。支持按 `R` 键重新测试，按 `ESC` 退出。

---

## 四、废弃模块（deprecated/）

这些是项目早期版本的代码，已被 `run/` 目录下的新代码替代。

| 文件 | 说明 |
|------|------|
| `main.py` | 旧版主程序：使用DFS随机生成迷宫，支持键盘操作和鼠标触发随机AI |
| `main_new.py` | 旧版主程序：使用自定义大地图，纯键盘操作 |
| `main_new1.py` | 旧版主程序：使用6×6小地图，纯键盘操作 |
| `maze.py` | 旧版迷宫生成器：`room` 类实现DFS随机迷宫生成算法 |
| `mapp1.py` | 旧版6×6小地图数据 |
| `q_agent.py` | 旧版Q-Learning智能体：基于Q表的经典Q-Learning（已被DQN替代） |

### 旧版 Q-Learning vs 当前 DQN 对比

| 特性 | Q-Learning（旧版） | DQN（当前） |
|------|------|------|
| 状态表示 | 离散状态编号 | 归一化连续坐标 |
| 值函数存储 | Q表（numpy数组） | 神经网络（4层FC） |
| 泛化能力 | 无（只能查表） | 有（相似状态可泛化） |
| 经验回放 | 无 | 有（ReplayBuffer） |
| 双网络 | 无 | 有（policy + target） |
| 适用规模 | 小地图（6×6） | 大地图（43×42） |

---

## 五、依赖关系

### 5.1 外部依赖

| 包 | 版本要求 | 用途 |
|------|------|------|
| `torch` | ≥2.0.0 | DQN神经网络构建与训练 |
| `torchvision` | - | PyTorch视觉工具包（依赖项） |
| `torchaudio` | - | PyTorch音频工具包（依赖项） |
| `numpy` | - | 数组操作、状态表示 |
| `pygame` | - | 游戏窗口、图形渲染、事件处理 |
| `matplotlib` | - | 训练reward曲线绘制 |

**GPU支持**：项目支持 CUDA 加速训练，会自动检测并使用 GPU。

### 5.2 模块依赖图

```
main_ai.py
├── maze_env.py
│   ├── mapp.py (地图数据)
│   └── score_map.py (BFS分数矩阵)
├── dqn_agent.py
│   ├── dqn/dqn_net.py (DQN网络)
│   └── dqn/replay_buffer.py (经验回放)
├── mapp.py (地图数据，用于渲染)
└── color.py (颜色常量)

test_movement.py
└── maze_env.py → mapp.py, score_map.py

test_movement_gui.py
├── maze_env.py → mapp.py, score_map.py
├── mapp.py
└── color.py

manhattan_distance.py
└── mapp.py (生成score_matrix的源数据)
```

---

## 六、项目运行方式

### 6.1 环境准备

```bash
# 安装依赖（CPU版本）
pip install torch torchvision torchaudio numpy pygame matplotlib

# 安装依赖（GPU版本，CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pygame matplotlib
```

### 6.2 启动主程序

```bash
cd run/
python main_ai.py
```

**运行流程**：
1. 程序启动后自动检测 GPU/CPU
2. 检查 `run/statistic/dqn_model.pth` 是否存在
   - **存在**：加载已有模型，直接进入菜单
   - **不存在**：开始500轮DQN训练，训练完成后保存模型和reward曲线
3. 显示主菜单：
   - 按 `1`：AI自动演示模式
   - 按 `2`：手动操作模式（方向键移动）
   - 按 `Q`：退出

### 6.3 运行测试

```bash
# 命令行版移动测试
cd run/
python test_movement.py

# GUI版移动测试
cd run/
python test_movement_gui.py
```

### 6.4 重新生成分数矩阵

```bash
cd run/
python manhattan_distance.py
```

输出可直接复制到 `score_map.py` 中替换 `score_matrix`。

---

## 七、关键算法说明

### 7.1 DQN算法核心

本项目使用的是经典的 **DQN** 算法，主要特点：

1. **双网络结构**：策略网络（policy_net）和目标网络（target_net），目标网络每200步从策略网络同步一次权重
2. **经验回放**：使用容量为10000的回放池，每次随机采样64条经验进行训练，打破数据相关性
3. **ε-greedy探索**：初始ε=1.0，每轮训练后乘以0.99衰减，最低降至0.01
4. **Huber损失**：使用 `SmoothL1Loss` 替代MSE，对异常值更鲁棒

### 7.2 奖励塑形（Reward Shaping）

这是项目的一个关键创新点。通过 BFS 预计算每个位置到终点的最短步数，在 `step()` 中计算潜在奖励差：

- **靠近终点**（`potential_old > potential_new`）：正奖励，鼓励继续靠近
- **远离终点**（`potential_old < potential_new`）：加倍负惩罚，强烈抑制远离行为
- **撞墙**：固定 -2 惩罚
- **到达终点**：+100 大奖励

这种塑形方式使得稀疏奖励问题变得稠密，极大加速了训练收敛。

### 7.3 状态表示

状态为归一化的二维坐标 `[x / (W-1), y / (H-1)]`，将离散的网格坐标映射到 [0, 1] 范围内的连续值，便于神经网络处理。

---

## 八、地图数据编码规范

| 值 | 含义 | 颜色渲染 |
|------|------|------|
| `0` | 通路 | 白色矩形边框 |
| `1` | 墙壁 | 黑色实心矩形 |
| `3` | 终点 | 红色圆点（渲染后变为通路） |
| `9` | 起点 | 蓝色圆点（渲染后变为通路） |

**注意**：起点和终点在渲染后会被重置为通路（0），但在 `MazeEnv` 中通过 `start_pos` 和 `goal_pos` 属性保留其位置信息。

---

## 九、训练输出

训练过程会在 `run/statistic/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `dqn_model.pth` | PyTorch模型权重文件，包含策略网络参数 |
| `reward_curve.png` | 训练过程中的reward变化曲线图 |

训练过程中每轮会在控制台输出：
```
Episode {n}/{total}, Reward: {r}, Steps: {s}, Epsilon: {e}
```

每50轮会在Pygame窗口中显示训练进度。

---

## 十、项目演进历史

| 日期 | 里程碑 |
|------|------|
| 初始 | 基础迷宫游戏（DFS随机生成 + 键盘/鼠标AI） |
| 后续 | 自定义大地图迷宫 + 手动操作模式 |
| 2026/4/12 | 完成Q-Learning强化学习与小地图游戏对接 |
| 2026/4/13 | 实现DQN学习、自动绘制reward曲线、保存训练模型 |
| 2026/4/14 | 实现大地图展示（DQN学习不稳定） |
| 2026/4/22 | 通过BFS生成最短路径分数矩阵，用于奖励塑形 |
| 2026/4/24 | 调整参数后AI训练中开始到达终点（训练仍不稳定） |
| 2026/5/9 | 修改了势函数奖励和禁用BatchNorm让AI稳定训练找到最短路径 |