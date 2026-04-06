import cv2
import numpy as np

class EnemyDetector:
    def __init__(self):
        # --- 1. 定义深青色 (车身主体) ---
        self.lower_cyan = np.array([85, 80, 30])
        self.upper_cyan = np.array([105, 255, 100])

        # --- 2. 定义浅灰色 (履带部分) ---
        self.lower_gray = np.array([0, 0, 130])
        self.upper_gray = np.array([180, 60, 200])

        # --- 3. 定义白色 (炮管高光/细节) ---
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])

    def get_mask(self, hsv):
        
        mask_cyan = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        mask_gray = cv2.inRange(hsv, self.lower_gray, self.upper_gray)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        # --- 4. 合并所有掩膜，使用按位或运算，将三个掩膜合二为一 ---
        mask = cv2.bitwise_or(mask_cyan, mask_gray)
        mask = cv2.bitwise_or(mask, mask_white)

        # 去除噪点，让检测更稳定
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # 开运算去噪
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 闭运算填充空洞
        return mask