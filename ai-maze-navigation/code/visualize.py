# visualize.py

import matplotlib.pyplot as plt
import numpy as np

def plot_path(env, agent):
    grid = np.zeros((env.size, env.size))

    for obs in env.obstacles:
        grid[obs] = -1

    grid[env.goal] = 2

    state = env.reset()
    path = [env.agent_pos]

    for _ in range(50):
        action = np.argmax(agent.q_table[state])
        state, _, done = env.step(action)
        path.append(env.agent_pos)

        if done:
            break

    for (x, y) in path:
        grid[x][y] = 1

    fig, ax = plt.subplots()

    im = ax.imshow(grid)

    # 关键1：设置刻度
    ax.set_xticks(np.arange(env.size))
    ax.set_yticks(np.arange(env.size))

    # 关键2：设置网格线（格子边界）
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)

    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

    # 去掉主网格
    ax.grid(False)

    # 显示坐标
    ax.set_xticklabels(range(env.size))
    ax.set_yticklabels(range(env.size))

    for i in range(len(path)-1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]

        dx = y2 - y1
        dy = x2 - x1

        ax.arrow(y1, x1, dx*0.3, dy*0.3, head_width=0.2, color='red')

    print("Final Path:")
    print(path)

    plt.title("Path Planning Result")
    plt.colorbar(im)
    plt.show()