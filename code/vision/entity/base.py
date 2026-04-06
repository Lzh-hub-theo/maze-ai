import cv2
import numpy as np

class BaseDetector:
    def __init__(self):
        self.tolerance = 5
        upper_limit = np.array([180, 255, 255])
        
        self.color1 = np.array([0, 0, 99])
        self.color2 = np.array([2, 255, 107])
        
        self.color_ranges = []
        for color in [self.color1, self.color2]:
            lower = np.clip(color - self.tolerance, 0, upper_limit)
            upper = np.clip(color + self.tolerance, 0, upper_limit)
            self.color_ranges.append((lower, upper))

        self.exclude_color = np.array([14, 255, 156])
        self.exclude_lower = np.clip(self.exclude_color - self.tolerance, 0, upper_limit)
        self.exclude_upper = np.clip(self.exclude_color + self.tolerance, 0, upper_limit)

    def get_mask(self, hsv):
        # 初始化一个全黑的掩膜
        mask_combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges:
            mask_current = cv2.inRange(hsv, lower, upper)
            mask_combined = cv2.bitwise_or(mask_combined, mask_current)

        mask_exclude = cv2.inRange(hsv, self.exclude_lower, self.exclude_upper)
        mask_combined = cv2.subtract(mask_combined, mask_exclude)

        kernel = np.ones((3,3), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        
        return mask_combined
