import cv2
import numpy as np

class RiverDetector:
    def __init__(self):
        # --- 1. 定义深蓝色 (河流暗部) ---
        # H: 100-120 (蓝色区间)
        # S: 150-255 (高饱和度)
        # V: 20-100 (低亮度)
        self.lower_dark_blue = np.array([100, 150, 20])
        self.upper_dark_blue = np.array([120, 255, 100])

        # --- 2. 定义浅蓝色 (河流亮部/浅像素) ---
        # H: 95-125 (蓝色区间，稍宽)
        # S: 40-140 (中低饱和度，偏淡)
        # V: 140-255 (高亮度)
        self.lower_light_blue = np.array([95, 40, 140])
        self.upper_light_blue = np.array([125, 140, 255])

    def get_mask(self, hsv):
        # --- 3. 分别创建掩膜 ---
        mask_dark = cv2.inRange(hsv, self.lower_dark_blue, self.upper_dark_blue)
        mask_light = cv2.inRange(hsv, self.lower_light_blue, self.upper_light_blue)

        # --- 4. 合并掩膜 ---
        # 使用按位或运算，将深蓝和浅蓝区域合并
        mask_river = cv2.bitwise_or(mask_dark, mask_light)

        # --- 5. 后处理 (可选) ---
        # 图片看起来是像素风，噪点可能较多，建议使用闭运算连接邻近像素
        kernel = np.ones((3,3), np.uint8)
        # 闭运算：先膨胀后腐蚀，用于填充物体内部的小空洞或连接邻近物体
        mask_river = cv2.morphologyEx(mask_river, cv2.MORPH_CLOSE, kernel)
        
        # 如果需要去除极小的噪点，可以加一个开运算
        # mask_river = cv2.morphologyEx(mask_river, cv2.MORPH_OPEN, kernel)

        return mask_river
