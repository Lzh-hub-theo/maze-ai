#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 测试文件：验证AI操作玩家移动的规则

import maze_env

# 动作映射：0=上 1=下 2=左 3=右
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_UP = 0
ACTION_DOWN = 1

def test_movement_sequence():
    """测试移动序列：左1步 -> 上2步 -> 下20步 -> 左2步 -> 右2步"""
    env = maze_env.MazeEnv()

    print("=" * 50)
    print("迷宫移动规则测试")
    print("=" * 50)
    print(f"起点位置: {env.start_pos}")
    print(f"终点位置: {env.goal_pos}")
    print(f"地图尺寸: {env.maze_width} x {env.maze_height}")
    print()

    # 重置环境，获取初始状态
    state = env.reset()
    print(f"初始位置: {env.agent_pos}")
    print()

    # 定义动作序列
    movements = [
        ("向左走1步", ACTION_LEFT, 1),
        ("向上走2步", ACTION_UP, 2),
        ("向下走20步", ACTION_DOWN, 20),
        ("向左走2步", ACTION_LEFT, 2),
        ("向右走2步", ACTION_RIGHT, 2),
    ]

    total_steps = 0
    for action_name, action, count in movements:
        print(f"--- {action_name} ---")
        for i in range(count):
            pos_before = env.agent_pos
            state, reward, done = env.step(action)
            pos_after = env.agent_pos
            total_steps += 1

            if pos_before == pos_after:
                print(f"  步骤{total_steps}: 位置{pos_before} -> 无法移动(撞墙或边界)")
            else:
                print(f"  步骤{total_steps}: 位置{pos_before} -> {pos_after}, 奖励: {reward:.4f}")

            if done:
                print(f"  到达终点！游戏结束")
                break

        print()

    print("=" * 50)
    print(f"测试完成！最终位置: {env.agent_pos}")
    print(f"总步数: {total_steps}")
    print("=" * 50)


if __name__ == '__main__':
    test_movement_sequence()
