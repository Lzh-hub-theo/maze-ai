#!/usr/bin/python3
# -*- coding: utf-8 -*-

from collections import deque
import mapp

# 原始迷宫矩阵（42行 × 43列）
map_list = mapp.map_list

def compute_distance_score(maze):
    """
    计算迷宫每个位置相对于起点(3)的“分数”
    距离越近分数越高，越远分数越低，不可达位置分数为0
    """
    rows = len(maze)
    cols = len(maze[0])
    
    # 1. 寻找起点坐标 (值为3)
    start = None
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 3:
                start = (r, c)
                break
        if start:
            break
    
    if not start:
        raise ValueError("迷宫中未找到起点（值为3的格子）")
    
    # 2. BFS 计算最短距离
    dist = [[-1] * cols for _ in range(rows)]   # -1 表示未访问/不可达
    q = deque()
    sr, sc = start
    dist[sr][sc] = 0
    q.append((sr, sc))
    
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
    
    # 3. 将距离映射为分数
    # 获取所有可达点的最大距离
    max_dist = max((d for row in dist for d in row if d != -1), default=0)
    
    score = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if dist[r][c] != -1:
                # 距离越小分数越高，最远点分数为1，起点分数为 max_dist + 1
                score[r][c] = max_dist - dist[r][c] + 1
            else:
                # 墙壁或不可达区域分数设为0
                score[r][c] = 0
                
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

