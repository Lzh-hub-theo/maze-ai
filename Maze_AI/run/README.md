# Maze AI - DQN 训练问题排查与解决

## 问题描述

在使用 DQN 进行迷宫探索强化学习时，训练过程中出现** Reward 崩溃现象**：
- 训练初期 Reward 逐渐改善
- 中期出现剧烈震荡
- 后期 Reward 急剧下降到 -1500 以下，智能体无法到达终点

## 问题分析

### 1. 根本原因：BatchNorm 在 RL 中不稳定

**问题代码** (`dqn/dqn_net.py`)：
```python
self.bn1 = nn.BatchNorm1d(hidden_dim)
self.bn2 = nn.BatchNorm1d(hidden_dim)
self.bn3 = nn.BatchNorm1d(hidden_dim)
```

BatchNorm 依赖 batch 统计量（mean/std），但 DQN 的训练数据来自 replay buffer 随机采样：
- **RL 中数据分布随训练变化**：policy 改变后，采样分布完全不同
- **batch size 只有 64**，统计量不稳定
- **训练/eval 模式切换时 BN 行为不同**，导致 Q 值估计不稳定

### 2. 辅助原因：缺少梯度裁剪

`dqn_agent.py` 中没有 `clip_grad_norm_`，梯度可能爆炸导致训练崩溃。

### 3. 训练频率过高

`main_ai.py` 中每步都调用 `agent.learn()`：
- 早期高 epsilon（≈1.0）时大量随机经验被反复训练
- 同一批数据被使用多次，造成灾难性遗忘
- replay buffer 被早期噪声主导

### 4. 状态表示过弱

`maze_env.py` 中的状态只有 2 维归一化坐标：
```python
def _get_state(self):
    x, y = self.agent_pos
    return np.array([x / (self.maze_width-1), y / (self.maze_height-1)], dtype=np.float32)
```

相邻格子的状态几乎相同（如 `(0.5, 0.3)` vs `(0.51, 0.3)`），网络无法有效区分迷宫中的不同位置。

## 解决方案

### 1. 移除 BatchNorm 和 Dropout

修改 `dqn/dqn_net.py`：
```python
class DQNNet(nn.Module):
    def __init__(self, input_dim, action_size, hidden_dim=256):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_size)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
```

### 2. 添加梯度裁剪

在 `dqn_agent.py` 的 `learn()` 方法中添加：
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
self.optimizer.step()
```

### 3. 降低训练频率（建议）

每个 episode 最多 learn 5-10 次，而不是每步都学：
```python
# 在 main_ai.py 中
if steps > 0 and steps % 100 == 0:  # 每 100 步学一次
    agent.learn()
```

### 4. 增强状态表示（可选）

可以加入附近墙壁的 one-hot 信息：
```python
def _get_state(self):
    x, y = self.agent_pos
    # 归一化坐标
    pos = np.array([x / (self.maze_width-1), y / (self.maze_height-1)], dtype=np.float32)
    # 周围墙壁信息
    walls = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        walls.append(1.0 if self._is_walkable(x+dx, y+dy) else 0.0)
    return np.concatenate([pos, walls])
```

## 关键参数参考

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-3 | 学习率 |
| gamma | 0.95 | 折扣因子 |
| epsilon_decay | 0.99 | 每轮衰减 1% |
| epsilon_min | 0.01 | 最小探索率 |
| target_update_freq | 200 | 目标网络更新频率 |
| batch_size | 64 | 批大小 |
| buffer_size | 10000 | 经验回放池大小 |

## 经验总结

1. **BatchNorm 在 RL 中通常不稳定**，建议使用 LayerNorm 或不使用 normalization
2. **梯度裁剪是 DQN 的标配**，可以防止梯度爆炸
3. **训练频率不宜过高**，每步都学习会导致早期噪声被过度强化
4. **状态表示要足够丰富**，否则网络无法区分相似状态
5. **epsilon 衰减要适中**，太慢会导致长期被噪声主导，太快会导致探索不足

## 致谢

问题排查过程中，通过分析 `result.log` 中的以下指标定位问题：
- `LearnStep`：学习步数，用于关联 target update 时机
- `BufferDoneRatio`：replay buffer 中完成 episode 的比例
- `Epsilon`：探索率变化
- `Steps`：每轮步数，辅助判断智能体行为