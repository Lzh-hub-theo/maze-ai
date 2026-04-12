#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Wonz
# 功能：带有强化学习的迷宫游戏 - AI自动演示和手动模式

import pygame
import sys
import random
import time
from pygame.locals import *
from random import randint, choice
import maze
import color
import mapp1
from maze_env import MazeEnv
from q_agent import QLearningAgent


# 常量定义
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
ROOM_SIZE = 15
TRAINING_EPISODES = 500  # 训练轮数
FONT_PATH = 'simhei.ttf'

# 动作映射：0=上 1=下 2=左 3=右
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_TO_DIRECTION = {
    0: (0, -1),  # 上
    1: (0, 1),   # 下
    2: (-1, 0),  # 左
    3: (1, 0)    # 右
}


def print_text(font, x, y, text, color_val, shadow=True):
    """输出文本信息"""
    if shadow:
        imgText = font.render(text, True, (0, 0, 0))
        screen.blit(imgText, (x-2, y-2))
    imgText = font.render(text, True, color_val)
    screen.blit(imgText, (x, y))


def train_agent(env, agent, episodes=TRAINING_EPISODES):
    """训练Q-learning agent"""
    print(f"开始训练Agent，共{episodes}轮次...")
    font = pygame.font.Font(FONT_PATH, 24)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 100:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # 每50轮打印一次训练进度
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            # 显示训练进度
            screen.fill(color.White)
            print_text(font, 300, 350, f"Training: {episode+1}/{episodes}", color.Black)
            pygame.display.flip()
            pygame.time.delay(100)
    
    print("训练完成！")


def action_to_room_change(roomx, roomy, action):
    """
    根据动作计算新的房间位置
    动作：0=上 1=下 2=左 3=右
    返回：(new_roomx, new_roomy)
    """
    dx, dy = ACTION_TO_DIRECTION.get(action, (0, 0))
    return (roomx + dx, roomy + dy)


def is_valid_move(roomx, roomy, action, r_list):
    """检查移动是否有效（不撞墙）"""
    new_roomx, new_roomy = action_to_room_change(roomx, roomy, action)
    
    # 检查边界
    if new_roomx < 0 or new_roomx >= 6 or new_roomy < 0 or new_roomy >= 6:
        return False
    
    # 检查是否是墙（值为1）
    return r_list[new_roomy][new_roomx] != 1


def play_ai_mode(env, agent, r_list):
    """AI自动演示模式"""
    print("进入AI自动演示模式...")
    screen.fill(color.White)
    
    # 加载角色
    user = pygame.image.load("user.png").convert_alpha()
    user = pygame.transform.smoothscale(user, (8, 8))
    
    # 绘制迷宫
    for i in range(6):
        for j in range(6):
            if r_list[j][i] == 0 or r_list[j][i] == 3:
                pygame.draw.rect(screen, color.White, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 1)
            elif r_list[j][i] == 1:
                pygame.draw.rect(screen, color.Black, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 0)
    
    # 绘制起点和终点
    pygame.draw.circle(screen, color.Blue, [35 + 5 * ROOM_SIZE, 35 + 2 * ROOM_SIZE], 5, 0)
    pygame.draw.circle(screen, color.Red, [35 + 0 * ROOM_SIZE, 35 + 2 * ROOM_SIZE], 5, 0)
    pygame.display.flip()
    
    # AI演示
    state = env.reset()
    roomx, roomy = 5, 2
    x = 30 + roomx * ROOM_SIZE
    y = 30 + roomy * ROOM_SIZE
    
    screen.blit(user, (x, y))
    pygame.display.flip()
    
    font = pygame.font.Font(FONT_PATH, 32)
    steps = 0
    max_steps = 100
    
    print("AI开始演示迷宫路径...")
    
    while steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        
        # AI选择动作（使用贪心策略）
        action = agent.get_best_action(state)
        
        # 尝试执行动作
        if is_valid_move(roomx, roomy, action, r_list):
            next_state, reward, done = env.step(action)
            
            # 更新位置
            dx, dy = ACTION_TO_DIRECTION[action]
            roomx += dx
            roomy += dy
            x = 30 + roomx * ROOM_SIZE
            y = 30 + roomy * ROOM_SIZE
            
            state = next_state
            steps += 1
            
            # 清除旧位置，绘制新位置
            screen.fill(color.White, (x - ROOM_SIZE, y, 10, 10) if dx > 0 else 
                                      (x + ROOM_SIZE, y, 10, 10) if dx < 0 else
                                      (x, y - ROOM_SIZE, 10, 10) if dy > 0 else
                                      (x, y + ROOM_SIZE, 10, 10))
            screen.blit(user, (x, y))
            
            # 显示步数
            screen.fill(color.White, (25, 0, 200, 25))
            print_text(font, 25, 0, f"Steps: {steps}", color.Black)
            pygame.display.flip()
            
            if done:
                # 到达目标
                screen.fill(color.White, (22, 0, 400, 30))
                print_text(font, 300, 0, f"AI Win! Steps: {steps}", color.Red)
                print_text(font, 250, 350, "按任意键继续", color.Black)
                pygame.display.flip()
                
                # 等待用户输入
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return False
                        elif event.type == pygame.KEYDOWN:
                            return True
            
            pygame.time.delay(500)  # 延迟500ms，便于观看
        else:
            # 如果当前动作无效，AI重新选择
            state = env._get_state()
    
    return True


def play_manual_mode(r_list):
    """手动模式 - 用户操作"""
    print("进入手动操作模式...")
    screen.fill(color.White)
    
    # 加载角色
    user = pygame.image.load("user.png").convert_alpha()
    user = pygame.transform.smoothscale(user, (8, 8))
    
    # 绘制迷宫
    for i in range(6):
        for j in range(6):
            if r_list[j][i] == 0 or r_list[j][i] == 3:
                pygame.draw.rect(screen, color.White, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 1)
            elif r_list[j][i] == 1:
                pygame.draw.rect(screen, color.Black, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 0)
    
    # 绘制起点和终点
    pygame.draw.circle(screen, color.Blue, [35 + 5 * ROOM_SIZE, 35 + 2 * ROOM_SIZE], 5, 0)
    pygame.draw.circle(screen, color.Red, [35 + 0 * ROOM_SIZE, 35 + 2 * ROOM_SIZE], 5, 0)

    # 初始位置
    roomx, roomy = 5, 2
    x = 30 + roomx * ROOM_SIZE
    y = 30 + roomy * ROOM_SIZE
    
    screen.blit(user, (x, y))
    pygame.display.flip()
    
    font = pygame.font.Font(FONT_PATH, 32)
    steps = 0
    gameover = False
    
    print("使用方向键移动，ESC返回菜单")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                
                # 检查目标
                if roomx == 0 and roomy == 2:
                    font = pygame.font.Font(FONT_PATH, 32)
                    print_text(font, 250, 350, "You Win!", color.Red)
                    print_text(font, 200, 400, "按任意键回到菜单", color.Black)
                    pygame.display.flip()
                    gameover = True
                    break
                
                # 方向键控制
                if event.key == pygame.K_RIGHT:
                    if is_valid_move(roomx, roomy, 3, r_list):
                        x += ROOM_SIZE
                        roomx += 1
                        steps += 1
                        screen.fill(color.White, (x - ROOM_SIZE, y, 10, 10))
                        screen.blit(user, (x, y))
                    else:
                        screen.fill(color.White, (300, 0, 200, 25))
                        print_text(font, 350, 0, "Wall!", color.Black)
                    
                elif event.key == pygame.K_LEFT:
                    if is_valid_move(roomx, roomy, 2, r_list):
                        x -= ROOM_SIZE
                        roomx -= 1
                        steps += 1
                        screen.fill(color.White, (x + ROOM_SIZE, y, 10, 10))
                        screen.blit(user, (x, y))
                    else:
                        screen.fill(color.White, (300, 0, 200, 25))
                        print_text(font, 350, 0, "Wall!", color.Black)
                
                elif event.key == pygame.K_UP:
                    if is_valid_move(roomx, roomy, 0, r_list):
                        y -= ROOM_SIZE
                        roomy -= 1
                        steps += 1
                        screen.fill(color.White, (x, y + ROOM_SIZE, 10, 10))
                        screen.blit(user, (x, y))
                    else:
                        screen.fill(color.White, (300, 0, 200, 25))
                        print_text(font, 350, 0, "Wall!", color.Black)
                
                elif event.key == pygame.K_DOWN:
                    if is_valid_move(roomx, roomy, 1, r_list):
                        y += ROOM_SIZE
                        roomy += 1
                        steps += 1
                        screen.fill(color.White, (x, y - ROOM_SIZE, 10, 10))
                        screen.blit(user, (x, y))
                    else:
                        screen.fill(color.White, (300, 0, 200, 25))
                        print_text(font, 350, 0, "Wall!", color.Black)
                
                # 更新步数显示
                screen.fill(color.White, (25, 0, 200, 25))
                print_text(font, 25, 0, f"Steps: {steps}", color.Black)
                pygame.display.flip()
        
        if gameover:
            # 等待用户输入
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        return True
        
        pygame.display.update()


def show_menu():
    """显示菜单"""
    screen.fill(color.White)
    font_large = pygame.font.Font(FONT_PATH, 48)
    font_small = pygame.font.Font(FONT_PATH, 32)
    
    print_text(font_large, 200, 150, "Maze AI Game", color.Black)
    print_text(font_small, 150, 300, "1. AI Auto Play (AI自动演示)", color.Black)
    print_text(font_small, 150, 350, "2. Manual Play (手动操作)", color.Black)
    print_text(font_small, 150, 400, "Q. Quit (退出)", color.Black)
    
    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 'ai'
                elif event.key == pygame.K_2:
                    return 'manual'
                elif event.key == pygame.K_q:
                    return None


# 游戏主函数
if __name__ == '__main__':
    # 初始化 Pygame
    pygame.init()
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    pygame.display.set_caption('Maze_AI（带强化学习）——by Wonz')
    
    clock = pygame.time.Clock()
    fps = 20
    
    # 创建环境和agent
    env = MazeEnv()
    state_size = env.maze_width * env.maze_height  # 6*6 = 36
    action_size = env.action_size  # 4
    agent = QLearningAgent(state_size, action_size)
    
    # 训练agent
    print("\n========== 开始训练强化学习Agent ==========")
    train_agent(env, agent, TRAINING_EPISODES)
    print("========== 训练完成 ==========\n")
    
    # 加载迷宫
    r_list = [row[:] for row in mapp1.map_list]
    
    # 修改地图中的3为0（起点标记变为通路）
    for i in range(6):
        for j in range(6):
            if r_list[j][i] == 3:
                r_list[j][i] = 0
    
    # 菜单循环
    while True:
        choice = show_menu()
        
        if choice is None:
            pygame.quit()
            sys.exit()
        elif choice == 'ai':
            continue_game = play_ai_mode(env, agent, r_list)
            if not continue_game:
                pygame.quit()
                sys.exit()
        elif choice == 'manual':
            continue_game = play_manual_mode(r_list)
            if not continue_game:
                pygame.quit()
                sys.exit()
