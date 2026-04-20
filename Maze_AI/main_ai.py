#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Wonz
# 功能：带有强化学习的迷宫游戏 - AI自动演示和手动模式

import pygame
import sys
from pygame.locals import *
from random import randint, choice
import color
import mapp
from maze_env import MazeEnv
from dqn_agent import DQNAgent

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available, training on CPU")


# 常量定义
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
ROOM_SIZE = 15
TRAINING_EPISODES = 5000  # 训练轮数
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
    """训练DQN agent，并记录每轮reward，训练结束后绘制reward曲线图"""
    print(f"开始训练Agent，共{episodes}轮次...")
    font = pygame.font.Font(FONT_PATH, 24)
    rewards_list = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while steps < 1000:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break
        agent.decay_epsilon()  # 每个episode结束后衰减epsilon
        rewards_list.append(total_reward)
        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")
        # 每50轮打印一次训练进度
        if (episode + 1) % 50 == 0:
            # 显示训练进度
            screen.fill(color.White)
            print_text(font, 300, 350, f"Training: {episode+1}/{episodes}", color.Black)
            pygame.display.flip()
            pygame.time.delay(100)
    print("训练完成！")
    # 训练结束后绘制reward曲线
    try:
        import matplotlib.pyplot as plt
        import os
        save_dir = os.path.join(os.path.dirname(__file__), 'statistic')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.figure(figsize=(8, 5))
        plt.plot(rewards_list, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DQN Training Reward Curve')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'reward_curve.png')
        plt.savefig(save_path)
        plt.close()
        print(f"训练曲线已保存到: {save_path}")
    except ImportError:
        print("未检测到matplotlib库，无法绘制reward曲线。请先安装matplotlib。");
    except Exception as e:
        print(f"绘制reward曲线时出错: {e}")


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
    
    rows = len(r_list)
    cols = len(r_list[0])

    # 检查边界
    if new_roomx < 0 or new_roomx >= cols or new_roomy < 0 or new_roomy >= rows:
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
    rows = len(r_list)
    cols = len(r_list[0])

    for j in range(rows):
        for i in range(cols):
            if r_list[j][i] == 0 or r_list[j][i] == 3:
                pygame.draw.rect(screen, color.White, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 1)
            elif r_list[j][i] == 1:
                pygame.draw.rect(screen, color.Black, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 0)
    
    # AI演示
    state = env.reset()
    endx, endy = end_pos

    # 绘制起点和终点
    roomx, roomy = env.agent_pos
    pygame.draw.circle(screen, color.Blue, [35 + roomx * ROOM_SIZE, 35 + roomy * ROOM_SIZE], 5, 0)
    pygame.draw.circle(screen, color.Red, [35 + endx * ROOM_SIZE, 35 + endy * ROOM_SIZE], 5, 0)
    pygame.display.flip()

    x = 30 + roomx * ROOM_SIZE
    y = 30 + roomy * ROOM_SIZE
    screen.blit(user, (x, y))
    pygame.display.flip()

    font = pygame.font.Font(FONT_PATH, 32)
    steps = 0
    max_steps = 3000

    print("AI开始演示迷宫路径...")

    while steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True

        action = agent.get_best_action(state)
        next_state, reward, done = env.step(action)
        # 直接用环境的agent_pos
        roomx, roomy = env.agent_pos
        x = 30 + roomx * ROOM_SIZE
        y = 30 + roomy * ROOM_SIZE
        state = next_state
        steps += 1

        # 清除旧位置，绘制新位置
        screen.fill(color.White)
        # 重新绘制迷宫
        for j in range(rows):
            for i in range(cols):
                if r_list[j][i] == 0 or r_list[j][i] == 3:
                    pygame.draw.rect(screen, color.White, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 1)
                elif r_list[j][i] == 1:
                    pygame.draw.rect(screen, color.Black, [30 + i * ROOM_SIZE, 30 + j * ROOM_SIZE, 10, 10], 0)
        pygame.draw.circle(screen, color.Blue, [35 + start_pos[0] * ROOM_SIZE, 35 + start_pos[1] * ROOM_SIZE], 5, 0)
        pygame.draw.circle(screen, color.Red, [35 + endx * ROOM_SIZE, 35 + endy * ROOM_SIZE], 5, 0)
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
        pygame.time.delay(500)
    return True


def play_manual_mode(r_list):
    """手动模式 - 用户操作"""
    print("进入手动操作模式...")
    screen.fill(color.White)
    
    # 加载角色
    user = pygame.image.load("user.png").convert_alpha()
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

    # 初始位置
    roomx, roomy = start_pos
    endx, endy = end_pos

    # 绘制起点和终点
    pygame.draw.circle(screen, color.Blue, [35 + roomx * ROOM_SIZE, 35 + roomy * ROOM_SIZE], 5, 0)
    pygame.draw.circle(screen, color.Red, [35 + endx * ROOM_SIZE, 35 + endy * ROOM_SIZE], 5, 0)

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
                if (roomx, roomy) == end_pos:
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
    print_text(font_small, 150, 300, "1. Maze AI Game (AI自动演示)", color.Black)
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
    action_size = env.action_size  # 4
    agent = DQNAgent(
        action_size,
        learning_rate=1e-3,   # 可调
        gamma=0.999,           # 统一gamma
        epsilon=1.0,          # 可调
        epsilon_decay=0.9995,  # 可调
        epsilon_min=0.01       # 可调
    )

    # DQN模型保存路径
    import os
    model_path = os.path.join(os.path.dirname(__file__), 'statistic', 'dqn_model.pth')

    # 检查模型是否存在，存在则加载，否则训练
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"检测到已训练模型，已加载DQN模型：{model_path}\n可直接进行AI演示或手动游戏。")
            trained = True
        except Exception as e:
            print(f"加载DQN模型失败，将重新训练。错误信息：{e}")
            trained = False
    else:
        trained = False

    if not trained:
        print("\n========== 开始训练DQN Agent ==========")
        train_agent(env, agent, TRAINING_EPISODES)
        print("========== 训练完成 ==========\n")
        # 保存模型
        save_dir = os.path.dirname(model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        agent.save(model_path)
        print(f"DQN模型已保存到：{model_path}")

    # 加载大地图
    r_list = [row[:] for row in mapp.map_list]
    # TODO 修改地图中的3为0（起点标记变为通路）
    start_pos = None
    end_pos = None

    for y in range(len(r_list)):
        for x in range(len(r_list[0])):
            if r_list[y][x] == 9:
                start_pos = (x, y)
                r_list[y][x] = 0
            elif r_list[y][x] == 3:
                end_pos = (x, y)

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
