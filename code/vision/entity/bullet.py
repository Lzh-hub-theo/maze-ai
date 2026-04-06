import cv2
import numpy as np

class BulletDetector:
    def __init__(self):
        # --- 1. 定义灰色 (子弹主体) ---
        self.lower_gray = np.array([0, 0, 140])
        self.upper_gray = np.array([180, 45, 255])

    def get_mask(self, hsv):
        # --- 2. 创建掩膜 ---
        mask = cv2.inRange(hsv, self.lower_gray, self.upper_gray)

        # --- 3. 后处理 (关键步骤) ---
        # 因为子弹很小，我们需要去除噪点，同时保留小物体
        kernel = np.ones((3, 3), np.uint8)
        
        # 开运算：先腐蚀后膨胀，去除细小的噪点（比如背景的纹理）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 闭运算：先膨胀后腐蚀，连接断裂的区域（防止子弹因为光照不均变成两半）
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask
