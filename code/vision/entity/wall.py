import cv2
import numpy as np

class WallDetector:
    def __init__(self):
        # --- 1. 定义橙红色 (砖块主体) ---
        # H: 0-25 (红色到橙色区间)
        # S: 100-255 (排除低饱和度的灰色)
        # V: 50-255 (包含暗红和亮橙)
        self.lower_orange_red = np.array([0, 100, 50])
        self.upper_orange_red = np.array([25, 255, 255])

        # --- 2. 定义深灰色 (砖块缝隙/阴影) ---
        # H: 0-180 (灰色无色相)
        # S: 0-60 (低饱和度)
        # V: 40-150 (中低亮度，区别于白色的亮光)
        self.lower_dark_gray = np.array([0, 0, 40])
        self.upper_dark_gray = np.array([180, 60, 150])

    def get_mask(self, hsv):
        # --- 3. 创建掩膜 ---
        mask_orange_red = cv2.inRange(hsv, self.lower_orange_red, self.upper_orange_red)
        mask_dark_gray = cv2.inRange(hsv, self.lower_dark_gray, self.upper_dark_gray)

        # --- 4. 合并掩膜 ---
        # 砖墙 = 砖块(橙红) + 缝隙(深灰)
        mask_wall = cv2.bitwise_or(mask_orange_red, mask_dark_gray)

        # --- 5. 后处理 ---
        # 砖墙通常是大面积连通的，使用闭运算填充砖块内部或缝隙间的小空洞
        kernel = np.ones((3,3), np.uint8)
        mask_wall = cv2.morphologyEx(mask_wall, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 可选：去除极小的噪点
        mask_wall = cv2.morphologyEx(mask_wall, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask_wall
