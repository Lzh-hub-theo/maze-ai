import cv2

class ShapeFilter:
    # 所有过滤器统一入口

    # 基础过滤（强烈推荐复用）
    @staticmethod
    def filter_by_area(contours, min_area=0, max_area=1e9):
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                result.append(cnt)
        return result

    @staticmethod
    def filter_by_aspect_ratio(contours, min_ratio=0.0, max_ratio=10.0):
        result = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            ratio = w / float(h)
            if min_ratio < ratio < max_ratio:
                result.append(cnt)
        return result

    @staticmethod
    def filter_by_extent(contours, min_extent=0.0):
        """
        extent = 轮廓面积 / 外接矩形面积 越接近1说明越“实心”
        """
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h

            if rect_area == 0:
                continue

            extent = area / rect_area

            if extent > min_extent:
                result.append(cnt)

        return result

    # 组合过滤（语义层）
    @staticmethod
    def filter_tank(contours):
        """
        坦克过滤规则
        """
        contours = ShapeFilter.filter_by_area(contours, 50, 2000)
        contours = ShapeFilter.filter_by_aspect_ratio(contours, 0.5, 2.0)
        contours = ShapeFilter.filter_by_extent(contours, 0.4)
        return contours

    @staticmethod
    def filter_wall(contours):
        """
        墙体过滤规则（通常更大、更方）
        """
        contours = ShapeFilter.filter_by_area(contours, 100, 10000)
        contours = ShapeFilter.filter_by_extent(contours, 0.5)
        return contours

    @staticmethod
    def filter_bullet(contours):
        """
        子弹（小 + 细）
        """
        contours = ShapeFilter.filter_by_area(contours, 5, 200)
        contours = ShapeFilter.filter_by_aspect_ratio(contours, 0.3, 3.0)
        return contours

    @staticmethod
    def debug_draw(img, contours, color=(0,255,0)):
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 1)