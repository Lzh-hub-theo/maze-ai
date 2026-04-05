import cv2

class ShapeFilter:

    @staticmethod
    def filter_tank(contours):
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 200 or area > 2500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h)

            # 1. 近似正方形
            if not (0.75 < ratio < 1.25):
                continue

            # 2. 面积填充率
            rect_area = w * h
            extent = area / rect_area

            if extent < 0.5:   
                continue

            result.append(cnt)

        return result

    @staticmethod
    def filter_bullet(contours):
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 80:
                result.append(cnt)
        return result

    @staticmethod
    def filter_wall(contours):
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                result.append(cnt)
        return result