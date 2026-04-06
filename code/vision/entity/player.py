import cv2
import numpy as np

class PlayerTankDetector:
    def __init__(self):
        # --- 1. 定义亮黄色 (坦克主体/高光) ---
        # H: 20-35 (标准黄色)
        # S: 100-255 (中高饱和度)
        # V: 150-255 (高亮度)
        self.lower_bright_yellow = np.array([20, 100, 150])
        self.upper_bright_yellow = np.array([35, 255, 255])

        # --- 2. 定义暗黄色/灰黄色 (坦克阴影/细节) ---
        # H: 15-35 (偏橙黄的褐色)
        # S: 50-150 (中低饱和度，带点灰)
        # V: 50-150 (中低亮度)
        self.lower_dark_yellow = np.array([15, 50, 50])
        self.upper_dark_yellow = np.array([35, 150, 150])

    def get_mask(self, hsv):
        # --- 3. 创建掩膜 ---
        mask_bright = cv2.inRange(hsv, self.lower_bright_yellow, self.upper_bright_yellow)
        mask_dark = cv2.inRange(hsv, self.lower_dark_yellow, self.upper_dark_yellow)

        # --- 4. 合并掩膜 ---
        # 使用按位或运算，将亮部和暗部合二为一
        mask_tank = cv2.bitwise_or(mask_bright, mask_dark)

        # --- 5. 后处理 ---
        # 去除噪点，让检测更稳定
        kernel = np.ones((3, 3), np.uint8)
        # 开运算：先腐蚀后膨胀，去除小白点噪点
        mask_tank = cv2.morphologyEx(mask_tank, cv2.MORPH_OPEN, kernel)
        # 闭运算：先膨胀后腐蚀，填充内部小黑洞
        mask_tank = cv2.morphologyEx(mask_tank, cv2.MORPH_CLOSE, kernel)

        return mask_tank
