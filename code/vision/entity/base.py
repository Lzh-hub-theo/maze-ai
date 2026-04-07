import cv2
import numpy as np
from ..filters import ShapeFilter

class BaseDetector:
    def __init__(self):
        pass

    # 指定位置生成掩膜
    def get_mask(self, hsv):
        height, width = hsv.shape[:2]
        
        mask = np.zeros((height, width), dtype=np.uint8)
        
        y1, x1 = 385, 194
        y2, x2 = 413, 224
        
        mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def detect_object(self, hsv, img):
        # 9. 基地
        mask_base = self.get_mask(hsv)
        contours_base, _ = cv2.findContours(mask_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_base = ShapeFilter.filter_wall(contours_base)
        for cnt in contours_base:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)  # 绿色
            cv2.putText(img, "base", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
