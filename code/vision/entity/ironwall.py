import cv2
import numpy as np


class IronWallDetector:
    def __init__(self):
        # --- 1. 定义白色 (铁墙中心正方形) ---
        # H: 0-180 (无色相)
        # S: 0-30 (极低饱和度)
        # V: 200-255 (极高亮度)
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])

        # --- 2. 定义灰色 (铁墙外框) ---
        # H: 0-180 (无色相)
        # S: 0-60 (低饱和度)
        # V: 100-190 (中等亮度，区别于白色和黑色)
        self.lower_gray = np.array([0, 0, 100])
        self.upper_gray = np.array([180, 60, 190])

    def get_mask(self, hsv):
        # --- 3. 创建掩膜 ---
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        mask_gray = cv2.inRange(hsv, self.lower_gray, self.upper_gray)

        # --- 4. 合并掩膜 ---
        # 铁墙 = 白色中心 + 灰色外框
        mask_wall = cv2.bitwise_or(mask_white, mask_gray)

        # --- 5. 后处理 ---
        # 开运算去除噪点，闭运算填充内部空洞
        kernel = np.ones((3, 3), np.uint8)
        mask_wall = cv2.morphologyEx(mask_wall, cv2.MORPH_OPEN, kernel)
        mask_wall = cv2.morphologyEx(mask_wall, cv2.MORPH_CLOSE, kernel)

        return mask_wall
