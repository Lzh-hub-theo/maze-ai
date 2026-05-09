#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 测试文件：验证AI操作玩家移动 - 图形界面版

import pygame
import sys
import color
import mapp
from maze_env import MazeEnv

# 常量定义
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
ROOM_SIZE = 15
FONT_PATH = "E:/Project/tank-battle/Maze_AI/run/simhei.ttf"
USER_PATH = "E:/Project/tank-battle/Maze_AI/run/user.png"

# 动作映射：0=上 1=下 2=左 3=右
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


def print_text(font, x, y, text, color_val, shadow=True):
    """输出文本信息"""
    if shadow:
        imgText = font.render(text, True, (0, 0, 0))
        screen.blit(imgText, (x-2, y-2))
    imgText = font.render(text, True, color_val)
    screen.blit(imgText, (x, y))


def test_fixed_sequence():
    """测试固定动作序列"""
    global screen

    # 初始化 Pygame
    pygame.init()
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    pygame.display.set_caption('Maze AI - 移动规则测试')
    clock = pygame.time.Clock()

    # 创建环境
    env = MazeEnv()
    r_list = [row[:] for row in mapp.map_list]

    # 获取起点和终点
    start_pos = None
    end_pos = None
    for y in range(len(r_list)):
        for x in range(len(r_list[0])):
            if r_list[y][x] == 9:
                start_pos = (x, y)
                r_list[y][x] = 0  # 起点标记变为通路
            elif r_list[y][x] == 3:
                end_pos = (x, y)

    # 加载角色图片
    user = pygame.image.load(USER_PATH).convert_alpha()
    user = pygame.transform.smoothscale(user, (8, 8))

    # 绘制迷宫
    rows = len(r_list)
    cols = len(r_list[0])

    for j in range(rows):
        for i in range(cols):
            if r_list[j][i] == 0 or r_list[j][i] == 3:
                pygame.draw.rect(screen, color.White, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 1)
            elif r_list[j][i] == 1:
                pygame.draw.rect(screen, color.Black, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 0)

    # 绘制起点和终点
    pygame.draw.circle(screen, color.Blue, [35 + start_pos[0] * ROOM_SIZE, 35 + start_pos[1] * ROOM_SIZE], 5, 0)
    pygame.draw.circle(screen, color.Red, [35 + end_pos[0] * ROOM_SIZE, 35 + end_pos[1] * ROOM_SIZE], 5, 0)

    # 重置环境
    state = env.reset()

    # 显示初始位置
    roomx, roomy = env.agent_pos
    x = 30 + roomx * ROOM_SIZE
    y = 30 + roomy * ROOM_SIZE
    screen.blit(user, (x, y))

    font = pygame.font.Font(FONT_PATH, 32)
    print_text(font, 25, 0, "Steps: 0", color.Black)
    print_text(font, 200, 0, "移动规则测试 - 固定序列", color.Black)
    pygame.display.flip()

    print("测试开始...")
    print(f"起点: {start_pos}, 终点: {end_pos}")

    # 定义动作序列: (动作名, 动作值, 次数)
    movements = [
        ("向左", ACTION_LEFT, 1),
        ("向上", ACTION_UP, 2),
        ("向下", ACTION_DOWN, 20),
        ("向左", ACTION_LEFT, 2),
        ("向右", ACTION_RIGHT, 2),
    ]

    total_steps = 0
    sequence_complete = False

    # 主循环
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_r and sequence_complete:
                    # 按R键重新开始测试
                    env.reset()
                    total_steps = 0
                    sequence_complete = False
                    # 重新绘制迷宫
                    screen.fill(color.White)
                    for j in range(rows):
                        for i in range(cols):
                            if r_list[j][i] == 0 or r_list[j][i] == 3:
                                pygame.draw.rect(screen, color.White, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 1)
                            elif r_list[j][i] == 1:
                                pygame.draw.rect(screen, color.Black, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 0)
                    pygame.draw.circle(screen, color.Blue, [35 + start_pos[0] * ROOM_SIZE, 35 + start_pos[1] * ROOM_SIZE], 5, 0)
                    pygame.draw.circle(screen, color.Red, [35 + end_pos[0] * ROOM_SIZE, 35 + end_pos[1] * ROOM_SIZE], 5, 0)
                    roomx, roomy = env.agent_pos
                    x = 30 + roomx * ROOM_SIZE
                    y = 30 + roomy * ROOM_SIZE
                    screen.blit(user, (x, y))
                    print_text(font, 25, 0, "Steps: 0", color.Black)
                    print_text(font, 200, 0, "移动规则测试 - 固定序列", color.Black)
                    pygame.display.flip()
                    print("测试重新开始...")

        if not sequence_complete:
            # 执行动作序列
            for action_name, action, count in movements:
                for i in range(count):
                    # 检查退出
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                    pos_before = env.agent_pos
                    state, reward, done = env.step(action)
                    pos_after = env.agent_pos
                    total_steps += 1

                    if pos_before == pos_after:
                        print(f"步骤{total_steps}: {action_name} -> 撞墙/边界")
                    else:
                        print(f"步骤{total_steps}: {action_name} -> {pos_before} -> {pos_after}")
                        # 更新玩家位置
                        roomx, roomy = env.agent_pos
                        x = 30 + roomx * ROOM_SIZE
                        y = 30 + roomy * ROOM_SIZE

                        # 重绘迷宫
                        screen.fill(color.White)
                        for j in range(rows):
                            for i in range(cols):
                                if r_list[j][i] == 0 or r_list[j][i] == 3:
                                    pygame.draw.rect(screen, color.White, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 1)
                                elif r_list[j][i] == 1:
                                    pygame.draw.rect(screen, color.Black, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 0)

                        pygame.draw.circle(screen, color.Blue, [35 + start_pos[0] * ROOM_SIZE, 35 + start_pos[1] * ROOM_SIZE], 5, 0)
                        pygame.draw.circle(screen, color.Red, [35 + end_pos[0] * ROOM_SIZE, 35 + end_pos[1] * ROOM_SIZE], 5, 0)
                        screen.blit(user, (x, y))

                        # 更新步数显示
                        screen.fill(color.White, (25, 0, 200, 25))
                        print_text(font, 25, 0, f"Steps: {total_steps}", color.Black)
                        print_text(font, 200, 0, f"正在执行: {action_name}", color.Black)
                        pygame.display.flip()

                        if done:
                            screen.fill(color.White, (400, 0, 400, 30))
                            print_text(font, 400, 0, "到达终点!", color.Red)
                            pygame.display.flip()
                            sequence_complete = True
                            break

                    pygame.time.delay(1000)  # 1000ms延迟

                if sequence_complete:
                    break

            if not sequence_complete:
                sequence_complete = True
                print(f"\n动作序列执行完毕! 最终位置: {env.agent_pos}")
                print("按R键重新测试，按ESC退出")

                # 清空动作显示区域
                screen.fill(color.White, (200, 0, 400, 30))
                print_text(font, 200, 0, "执行完毕! 按R重新测试", color.Black)
                pygame.display.flip()

        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    test_fixed_sequence()
