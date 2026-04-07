import cv2
import numpy as np
from ..partition import GridPartition

class RiverDetector:
    def __init__(self):
        self.tolerance = 2
        upper_limit = np.array([180, 255, 255])
        
        # --- np.clip 防止数值越界 ---
        c1 = np.array([90, 62, 239])
        self.lower_dark_blue = np.clip(c1 - self.tolerance, 0, upper_limit) # H上限180, S/V上限255
        self.upper_dark_blue = np.clip(c1 + self.tolerance, 0, upper_limit)

        c2 = np.array([120, 189, 255])
        self.lower_light_blue = np.clip(c2 - self.tolerance, 0, upper_limit)
        self.upper_light_blue = np.clip(c2 + self.tolerance, 0, upper_limit)

    def get_mask(self, hsv):
        # --- 创建掩膜 ---
        mask_dark = cv2.inRange(hsv, self.lower_dark_blue, self.upper_dark_blue)
        mask_light = cv2.inRange(hsv, self.lower_light_blue, self.upper_light_blue)

        # --- 合并掩膜 ---
        # 使用按位或运算，将深蓝和浅蓝区域合并
        mask_river = cv2.bitwise_or(mask_dark, mask_light)

        # 形态学操作：连接邻近的色块
        kernel = np.ones((3,3), np.uint8)
        mask_river = cv2.morphologyEx(mask_river, cv2.MORPH_CLOSE, kernel)

        return mask_river
    
    def detect_object(self, hsv, img):
        # 5. 水
        mask_water = self.get_mask(hsv)
        river_cells = GridPartition.extract_wall_cells(mask_water, grid_size=32)
        for (x, y, w, h) in river_cells:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 165, 255), 1)
            cv2.putText(img, "R", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
