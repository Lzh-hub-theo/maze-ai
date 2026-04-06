import cv2
import numpy as np

class GrassDetector:
    def __init__(self):
        # --- 1. 定义深绿色 (草丛暗部/阴影) ---
        # H: 35-85 (绿色区间)
        # S: 100-255 (中高饱和度)
        # V: 20-100 (低亮度)
        self.lower_dark_green = np.array([35, 100, 20])
        self.upper_dark_green = np.array([85, 255, 100])

        # --- 2. 定义浅绿色 (草丛亮部/高光) ---
        # H: 35-85 (绿色区间)
        # S: 40-255 (中低饱和度，防止把黄色识别进来)
        # V: 100-255 (中高亮度)
        self.lower_light_green = np.array([35, 40, 100])
        self.upper_light_green = np.array([85, 255, 255])

    def get_mask(self, hsv):
        # --- 3. 创建掩膜 ---
        mask_dark = cv2.inRange(hsv, self.lower_dark_green, self.upper_dark_green)
        mask_light = cv2.inRange(hsv, self.lower_light_green, self.upper_light_green)

        # --- 4. 合并掩膜 ---
        # 使用按位或运算，将深绿和浅绿的掩膜合二为一
        mask_grass = cv2.bitwise_or(mask_dark, mask_light)

        # --- 5. 后处理 ---
        # 由于图片是像素风格，可能会有噪点，使用形态学操作平滑
        kernel = np.ones((3,3), np.uint8)
        mask_grass = cv2.morphologyEx(mask_grass, cv2.MORPH_OPEN, kernel) # 开运算去噪
        mask_grass = cv2.morphologyEx(mask_grass, cv2.MORPH_CLOSE, kernel) # 闭运算填充内部空洞

        return mask_grass
