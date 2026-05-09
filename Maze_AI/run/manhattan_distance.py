#!/usr/bin/python3
# -*- coding: utf-8 -*-

from collections import deque
import mapp

# 原始迷宫矩阵（42行 × 43列）
map_list = mapp.map_list

def compute_distance_score(maze):
    """
    计算迷宫每个位置到终点(3)的最短步数
    起点(9)为坦克初始位置，终点(3)为最终目标
    每个格子的值 = 从这个点到终点(3)的最短步数
    """
    rows = len(maze)
    cols = len(maze[0])

    # 1. 寻找终点坐标 (值为3) - 作为BFS起点
    end = None
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 3:
                end = (r, c)
                break
        if end:
            break

    if not end:
        raise ValueError("迷宫中未找到终点（值为3的格子）")

    # 2. BFS 计算从终点(3)到每个位置的最短距离
    dist = [[-1] * cols for _ in range(rows)]   # -1 表示未访问/不可达
    q = deque()
    er, ec = end
    dist[er][ec] = 0
    q.append((er, ec))

    # 四个方向：上，下，左，右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # 墙壁(1)不可通过，其他(0,3,9)均视为可通行
                if maze[nr][nc] != 1 and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))

    # 3. 不可达位置分数设为很大的正数（agent不应该去那里）
    # 可达位置为到终点的步数
    max_reachable_dist = max(dist[r][c] for r in range(rows) for c in range(cols) if dist[r][c] != -1)
    INFINITE_DIST = max_reachable_dist + 100  # 比最远可达距离大100，确保不可达区域的shaping为负

    score = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if dist[r][c] == -1:
                # 不可达位置设为 INFINITE_DIST（很大的正数）
                # 当agent误入这些区域时，moving away会得到负reward
                score[r][c] = INFINITE_DIST
            else:
                score[r][c] = dist[r][c]

    return score

if __name__ == "__main__":
    result = compute_distance_score(map_list)

    # 计算整个矩阵中所有数字的最大位数（用于对齐）
    max_val = max(max(row) for row in result) + 1
    width = len(str(max_val))

    # 打印格式化的矩阵，每列宽度一致，右对齐
    print("score_matrix = [")
    for i, row in enumerate(result):
        # 将每个数字格式化为固定宽度的字符串
        # formatted_row = "[" + ", ".join(f"{val:{6}.1f}" for val in row) + "]"
        formatted_row = "[" + ", ".join(f"{val:{6}d}" for val in row) + "]"
        if i < len(result) - 1:
            print(f"    {formatted_row},")
        else:
            print(f"    {formatted_row}")
    print("]")