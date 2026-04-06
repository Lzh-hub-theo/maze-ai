import cv2
import numpy as np

class HSVTuner:
    def __init__(self):
        self.samples = []

    def mouse_callback(self, event, x, y, flags, param):
        img, hsv = param

        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = hsv[y, x]
            self.samples.append(pixel)
            print(f"采样 HSV: {pixel}")

    def run(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        clone = img.copy()
        cv2.namedWindow("Tuner")
        cv2.setMouseCallback("Tuner", self.mouse_callback, (clone, hsv))

        print("👉 点击目标区域采样，按 q 结束")

        while True:
            cv2.imshow("Tuner", clone)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(self.samples) == 0:
            print("没有采样")
            return None

        samples = np.array(self.samples)

        lower = np.percentile(samples, 5, axis=0)
        upper = np.percentile(samples, 95, axis=0)

        print("\n🎯 建议 HSV 范围：")
        print(f"lower = {lower}")
        print(f"upper = {upper}")

        return lower, upper