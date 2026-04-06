import cv2
import numpy as np

class GrassDetector:
    def __init__(self):
        self.tolerance = 2
        upper_limit = np.array([180, 255, 255])
        
        self.color1 = np.array([40, 255, 214])
        self.color2 = np.array([63, 255, 82])
        self.color3 = np.array([57, 255, 74])
        
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
