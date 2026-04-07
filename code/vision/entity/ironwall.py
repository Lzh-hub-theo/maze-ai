import cv2
import numpy as np
from ..partition import GridPartition

class IronWallDetector:
    def __init__(self):
        self.tolerance = 2
        upper_limit = np.array([180, 255, 255])
        
        self.color1 = np.array([0, 0, 255])
        self.color2 = np.array([0, 0, 173])
        self.color3 = np.array([120, 255, 255])
        
        self.color_ranges = []
        for color in [self.color1, self.color2, self.color3]:
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
        # 7. 铁墙
        mask_steel = self.get_mask(hsv)
        steel_cells = GridPartition.extract_wall_cells(mask_steel, grid_size=16)
        for (x, y, w, h) in steel_cells:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 165, 255), 1)
            cv2.putText(img, "S", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
