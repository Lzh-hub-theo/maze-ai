# visualize.py

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
def plot_path(env, agent):
    size = env.size

    # 定义颜色编码（用数字映射）
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4

    grid = np.zeros((size, size))

    # 设置障碍
    for obs in env.obstacles:
        grid[obs] = OBSTACLE

    # 设置起点和终点
    start_pos = (0, 0)
    grid[start_pos] = START
    grid[env.goal] = GOAL

    # 生成路径
    state = env.reset()
    path = [env.agent_pos]

    for _ in range(50):
        action = np.argmax(agent.q_table[state])
        state, _, done = env.step(action)
        path.append(env.agent_pos)
        if done:
            break

    # 标记路径（不覆盖起点终点）
    for (x, y) in path:
        if (x, y) != start_pos and (x, y) != env.goal:
            grid[x][y] = PATH

    # 🎨 自定义颜色（重点）
    color_map = {
        EMPTY: (0.8, 0.8, 0.8),   # 灰色（可移动区域）
        OBSTACLE: (0, 0, 0),      # 黑色
        START: (0, 0, 1),         # 蓝色
        GOAL: (1, 0, 0),          # 红色
        PATH: (0.6, 0.6, 0.6)     # 淡灰色
    }

    # 转成RGB图像
    rgb_grid = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            rgb_grid[i, j] = color_map[grid[i, j]]

    # ===== 绘图 =====
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.imshow(rgb_grid)

    # 网格线
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.grid(False)
    ax.set_title("路径规划结果")

    for i in range(len(path)-1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]

        dx = y2 - y1
        dy = x2 - x1

        ax.arrow(y1, x1, dx*0.1, dy*0.1, head_width=0.05, color='black')

    # ===== 右侧图例（重点）=====
    legend_items = [
        ("障碍物", (0, 0, 0)),
        ("可移动区域", (0.8, 0.8, 0.8)),
        ("起点", (0, 0, 1)),
        ("终点", (1, 0, 0)),
        ("最终路径", (0.6, 0.6, 0.6)),
    ]

    # 在右侧画图例
    for i, (label, color) in enumerate(legend_items):
        y_pos = 0.8 - i * 0.12

        # 小方块
        rect = Rectangle((1.05, y_pos), 0.05, 0.05,
                         transform=ax.transAxes,
                         color=color,
                         clip_on=False)

        ax.add_patch(rect)

        # 文本
        ax.text(1.12, y_pos + 0.02, label,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='center')

    plt.show()