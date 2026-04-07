import cv2
import numpy as np
from ..partition import GridPartition

class WallDetector:
    def __init__(self):
        self.tolerance = 2
        upper_limit = np.array([180, 255, 255])
        
        self.color1 = np.array([14, 255, 156])
        self.color2 = np.array([2, 255, 107])
        
        self.color_ranges = []
        for color in [self.color1, self.color2]:
            lower = np.clip(color - self.tolerance, 0, upper_limit)
            upper = np.clip(color + self.tolerance, 0, upper_limit)
            self.color_ranges.append((lower, upper))

    def get_mask(self, hsv):
        # 初始化一个全黑的掩膜
        mask_combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges:
            mask_current = cv2.inRange(hsv, lower, upper)
            mask_combined = cv2.bitwise_or(mask_combined, mask_current)

        kernel = np.ones((3,3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        
        return mask_combined
    
    def detect_object(self, hsv, img):
        # 4. 砖墙
        mask_brick = self.get_mask(hsv)
        brick_cells = GridPartition.extract_wall_cells(mask_brick, grid_size=8)
        for (x, y, w, h) in brick_cells:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 165, 255), 1)
            # cv2.putText(img, "Br", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
