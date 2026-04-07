import cv2
import numpy as np
from ..filters import ShapeFilter

class PlayerTankDetector:
    def __init__(self):
        self.tolerance = 2
        upper_limit = np.array([180, 255, 255])
        
        self.color1 = np.array([30, 92, 231])
        self.color2 = np.array([30, 255, 107])
        self.color3 = np.array([19, 219, 231])
        
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
    
    def detect_object(self, hsv, img, state):
        # 2. 玩家检测
        mask_player = self.get_mask(hsv)
        contours_player, _ = cv2.findContours(mask_player, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_player = ShapeFilter.filter_tank(contours_player)
        for cnt in contours_player:
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w//2, y + h//2)
            state['player_pos'] = center
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Player", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
