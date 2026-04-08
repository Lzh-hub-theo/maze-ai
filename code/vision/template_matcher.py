import cv2
import numpy as np

class TemplateMatcher:
    def __init__(self, template_paths, threshold=0.4):
        """
        template_paths: 模板图片路径列表
        threshold: 匹配阈值（0~1）
        """
        self.templates = []
        self.threshold = threshold

        for path in template_paths:
            tpl = cv2.imread(path, 0)  # 灰度
            if tpl is None:
                raise ValueError(f"模板加载失败: {path}")
            self.templates.append(tpl)

    def match(self, roi):
        """
        roi: 待检测区域（BGR）
        return: 是否匹配成功
        """
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best_score = 0

        for tpl in self.templates:
            # resize ROI 到模板大小（关键点！）
            resized = cv2.resize(roi_gray, (tpl.shape[1], tpl.shape[0]))

            res = cv2.matchTemplate(resized, tpl, cv2.TM_CCOEFF_NORMED)
            score = res.max()

            best_score = max(best_score, score)

        return best_score >= self.threshold, best_score