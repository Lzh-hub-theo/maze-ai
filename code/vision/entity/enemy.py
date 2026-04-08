import os
import cv2
import numpy as np
from ..filters import ShapeFilter
from ..template_matcher import TemplateMatcher

class EnemyDetector:
    def __init__(self):
        # 颜色HSV匹配
        self.tolerance = 2
        upper_limit = np.array([180, 255, 255])
        
        self.color1 = np.array([0, 0, 173])
        self.color2 = np.array([0, 0, 255])
        self.color3 = np.array([93, 255, 74])
        
        self.color_ranges = []
        for color in [self.color1, self.color2, self.color3]:
            lower = np.clip(color - self.tolerance, 0, upper_limit)
            upper = np.clip(color + self.tolerance, 0, upper_limit)
            self.color_ranges.append((lower, upper))

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # 模版匹配
        self.template_matcher = TemplateMatcher(
            template_paths=[
                os.path.join(BASE_DIR, "vision", "template", "enemy_bot", "tank1.png"),
                os.path.join(BASE_DIR, "vision", "template", "enemy_bot", "tank2.png"),
                os.path.join(BASE_DIR, "vision", "template", "enemy_bot", "tank3.png"),
                os.path.join(BASE_DIR, "vision", "template", "enemy_bot", "tank4.png")
            ]
        )

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
        # 3. 敌人检测
        mask_enemy = self.get_mask(hsv)
        contours_enemy, _ = cv2.findContours(mask_enemy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_enemy = ShapeFilter.filter_tank(contours_enemy) # 形状过滤，解决砖墙误判
        for cnt in contours_enemy:
            x, y, w, h = cv2.boundingRect(cnt)

            roi = img[y:y+h, x:x+w]
            is_enemy, score = self.template_matcher.match(roi)
            if not is_enemy:
                continue  # 过滤误检

            center = (x + w//2, y + h//2)
            state['enemy_positions'].append(center)
            state['enemy_count'] += 1
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, "Enemy", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
