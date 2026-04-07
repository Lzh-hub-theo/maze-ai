import cv2
import numpy as np
from ..filters import ShapeFilter

class BulletDetector:
    def __init__(self):
        # --- 1. 定义灰色 (子弹主体) ---
        self.lower_gray = np.array([0, 0, 140])
        self.upper_gray = np.array([180, 45, 255])

    def get_mask(self, hsv):
        # --- 2. 创建掩膜 ---
        mask = cv2.inRange(hsv, self.lower_gray, self.upper_gray)

        # 去除噪点，同时保留小物体
        kernel = np.ones((3, 3), np.uint8)
        
        # 开运算：先腐蚀后膨胀，去除细小的噪点（比如背景的纹理）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 闭运算：先膨胀后腐蚀，连接断裂的区域（防止子弹因为光照不均变成两半）
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask
    
    def detect_object(self, hsv, img):
        # 6. 子弹
        mask_bullet = self.get_mask(hsv)
        contours_bullet, _ = cv2.findContours(mask_bullet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_bullet = ShapeFilter.filter_bullet(contours_bullet)
        for cnt in contours_bullet:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)  # 蓝色
            cv2.putText(img, "bullet", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
