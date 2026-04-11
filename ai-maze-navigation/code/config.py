# config.py

# 地图大小
GRID_SIZE = 5

# Q-learning参数
ALPHA = 0.1        # 学习率
GAMMA = 0.9        # 折扣因子
EPSILON = 1.0      # 初始探索率
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1

# 训练参数
EPISODES = 1000
MAX_STEPS = 100

# 奖励
REWARD_GOAL = 100
REWARD_OBSTACLE = -100
REWARD_STEP = -1